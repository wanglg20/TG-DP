# Portions of this file are adapted from the original CAV-MAE by Yuan Gong.

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import  Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn.functional import all_gather as dist_all_gather
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_to_return = None    
        if return_attn:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn_to_return = attn
            attn = self.attn_drop(attn)
            x = attn @ v
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn_to_return
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,  drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None, return_attn=False):
        if modality == 'a':
            nx1 = self.norm1_a(x)
            nx2 = self.norm2_a
        elif modality == 'v':
            nx1 = self.norm1_v(x)
            nx2 = self.norm2_v
        else:
            nx1 = self.norm1(x)
            nx2 = self.norm2

        if return_attn:
            y, attn = self.attn(nx1, return_attn=True)
        else:
            y = self.attn(nx1, return_attn=False)
            attn = None

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(nx2(x)))
        return (x, attn) if return_attn else x

# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class CAVMAE_DP(nn.Module):
    """ CAV-MAE with Dual-Path Teacher guided
    """
    def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, num_register_tokens=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False, 
                 cls_token=False, global_local_losses=False, total_frame=16, contrastive_heads=True, multi_ratio_masking=False, keep_register_tokens=False,
                 cls_full=False, probe_guide=False):
        super().__init__()
        print('TG-DP model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        self.multi_ratio_masking = multi_ratio_masking
        self.keep_register_tokens = keep_register_tokens
        print('Keep Register Tokens: ', self.keep_register_tokens)
        self.cls_full = cls_full
        self.probe_guide = probe_guide
        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.audio_length = audio_length
        self.patch_embed_a.num_patches = int((self.audio_length / 16) * (128 / 16))
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.num_register_tokens = num_register_tokens
        print('Number of Registers: {:d}'.format(self.num_register_tokens))
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.num_register_tokens * 2, embed_dim))

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # unified branch
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(12-modality_specific_depth)])
        self.contrastive_heads = contrastive_heads
        if self.contrastive_heads:
            self.constrative_head_audio = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(2)])
            self.constrative_head_visual = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(2)])
        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        # the decoder part
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.cls_token = cls_token
        if self.cls_token:
            print("Using CLS Token")
            self.cls_token_a = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cls_token_v = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.global_local_losses = global_local_losses
        if self.global_local_losses:
            print("Using Global and Local Losses")

        self.total_frame = total_frame
        print('Using {:d} frames'.format(self.total_frame))

        self.intermediate_outputs = {}

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def register_hooks(self, blocks, block_type):
        def hook_fn(m, i, o):
            self.intermediate_outputs[block_type + str(m)] = o
        for idx, block in enumerate(blocks):
            block.register_forward_hook(hook_fn)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))

        decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.cls_token:
            torch.nn.init.normal_(self.cls_token_a, std=.02)
            torch.nn.init.normal_(self.cls_token_v, std=.02)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, c, h, w, p=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def get_patch2cls_rank(self,
                       attn_map: torch.Tensor,
                       mode: str = "deterministic",
                       temperature: float = 1.0,
                       eps: float = 1e-12):
        """
        input:
            attn_map: torch.Tensor, [B, H, 1+N, 1+N]
            mode:
                - "deterministic": original behavior, sort by attention mean from high to low
                - "prob": probability-based sorting, using Gumbel-top-k sampling to generate a random permutation based on the attention distribution
        output:
            ranks: LongTensor, [B, N]
            probs: FloatTensor, [B, N]
        """
        assert attn_map.dim() == 4, f"Expected 4D tensor, got {attn_map.shape}"
        B, H, L, _ = attn_map.shape
        num_patches = L - 1  # remove cls

        # [B, H, N]
        patch2cls = attn_map[:, :, 1:, 0]

        #  [B, N]
        patch2cls_mean = patch2cls.mean(dim=1)

        logits = patch2cls_mean / max(temperature, eps)
        probs = F.softmax(logits, dim=-1)  # [B, N]

        if mode == "deterministic":
            ranks = torch.argsort(patch2cls_mean, dim=1, descending=True)
            return ranks, probs

        elif mode == "prob":
            g = -torch.log(-torch.log(torch.clamp(torch.rand_like(probs), min=eps, max=1.0-eps)))
            scores = torch.log(torch.clamp(probs, min=eps)) + g
            ranks = torch.argsort(scores, dim=1, descending=True)  # [B, N]
            return ranks, probs

        else:
            raise ValueError(f"Unknown mode: {mode}, expected 'deterministic' or 'prob'")
        


    def forward_encoder(self, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured', teacher_mode=False, rank_a=None, rank_v=None):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v
        if rank_a is None and rank_v is None:
            # by default, we always use unstructured masking
            if mask_mode == 'unstructured':
                a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
            else:
                a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

            # visual branch always use unstructured masking
            v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)
        else:
            a, mask_a, ids_restore_a = self.ranked_masking(a, rank_a, mask_ratio_a)
            v, mask_v, ids_restore_v = self.ranked_masking(v, rank_v, mask_ratio_v)

        batch_size = a.shape[0]


        # Append cls tokens
        if self.cls_token:
            cls_tokens_a = self.cls_token_a.expand(batch_size, -1, -1)
            cls_tokens_v = self.cls_token_v.expand(batch_size, -1, -1)

            a = torch.cat([cls_tokens_a, a], dim=1)
            v = torch.cat([cls_tokens_v, v], dim=1)

        # Append register tokens
        if self.num_register_tokens > 0:
            
            r_a = self.register_tokens[:self.num_register_tokens].unsqueeze(0).expand(batch_size, -1, -1)
            r_v = self.register_tokens[self.num_register_tokens:].unsqueeze(0).expand(batch_size, -1, -1)
            
            a = torch.cat([a, r_a], dim=1)
            v = torch.cat([v, r_v], dim=1)

        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        if self.num_register_tokens > 0 and not self.keep_register_tokens:
            # Remove register tokens, keeping the cls token, if present
            a = a[:, :-self.num_register_tokens, :]
            v = v[:, :-self.num_register_tokens, :]

        # Concatenate audio and visual tokens without cls tokens
        x = torch.cat((a[:, 1:] if self.cls_token else a, v[:, 1:] if self.cls_token else v), dim=1)

        # unified stream, shared blocks_u, but independent normalization layers
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)


        if teacher_mode:
            for blk in self.blocks_u:
                ca, attn_a = blk(a, 'a', return_attn=True)    
            for blk in self.blocks_u:
                cv, attn_v = blk(v, 'v', return_attn=True)
            ca = self.norm_a(ca[:, 0, :].squeeze())
            cv = self.norm_v(cv[:, 0, :].squeeze())
            return attn_a, attn_v, ca, cv
        else:
            for blk in self.blocks_u:
                ca = blk(a, 'a')
            for blk in self.blocks_u:
                cv = blk(v, 'v')
        
        if self.contrastive_heads:
            for blk in self.constrative_head_audio:
                ca = blk(ca)
            for blk in self.constrative_head_visual:
                cv = blk(cv)

        # Remove register tokens, keeping the cls token, if present
        if self.num_register_tokens > 0 and self.keep_register_tokens:
            ca = ca[:, :-self.num_register_tokens, :]
            cv = cv[:, :-self.num_register_tokens, :]
            # To remove from x, we have to remove from where it was concatenated and at the end
            x = torch.cat((x[:,:self.patch_embed_a.num_patches, :], x[:,self.patch_embed_a.num_patches+self.num_register_tokens:-self.num_register_tokens, :]), dim=1)

        if self.cls_token:
            # split the local patch tokens from the cls tokens
            #cls tokens
            cls_a = self.norm_a(ca[:, 0, :].squeeze())  # This gets the audio CLS token from ca
            cls_v = self.norm_v(cv[:, 0, :].squeeze())  # This gets the visual CLS token from cv
            # local patch tokens
            ca = self.norm_a(ca[:, 1:, :])
            cv = self.norm_v(cv[:, 1:, :])

            return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv, cls_a, cls_v
        else:
            ca = self.norm_a(ca)
            cv = self.norm_v(cv)

        return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):

        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # no cls token
        a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # similar for the visual modality
        mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)  # no cls token
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # concatenate audio and visual tokens
        x = torch.cat([a_, v_], dim=1)

        decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
        x = x + decoder_pos_embed

        # add modality indication tokens
        x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
        x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_a = self.decoder_pred_a(x[:, :self.patch_embed_a.num_patches, :])
        x_v = self.decoder_pred_v(x[:, self.patch_embed_a.num_patches:, :])

        # return audio and video tokens
        return x_a, x_v

    def forward_contrastive_all_gather(self, audio_rep, video_rep, bidirect_contrast=False, mode='train', get_sample=False):
        audio_rep_local = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep_local = torch.nn.functional.normalize(video_rep, dim=-1)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        B = audio_rep_local.size(0)
        device = audio_rep_local.device
        temp = 0.05  

        if mode == 'train':
            gathered_v = dist_all_gather(video_rep_local)        # list[Tensor[B,D]]， rank0..rankN-1
            v_global = torch.cat(gathered_v, dim=0)      # [B*world, D]
            logits_av = (audio_rep_local @ v_global.t()) / temp  # [B, B*world]
            targets_av = rank * B + torch.arange(B, device=device)  
            loss_av = F.cross_entropy(logits_av, targets_av)

            gathered_a = dist_all_gather(audio_rep_local)
            a_global = torch.cat(gathered_a, dim=0)      # [B*world, D]
            logits_va = (video_rep_local @ a_global.t()) / temp  # [B, B*world]
            targets_va = rank * B + torch.arange(B, device=device)
            loss_va = F.cross_entropy(logits_va, targets_va)

            nce = 0.5 * (loss_av + loss_va)
    
            with torch.no_grad():
                acc_av = (logits_av.argmax(dim=1) == targets_av).float().mean()
                acc_va = (logits_va.argmax(dim=1) == targets_va).float().mean()
                c_acc = 0.5 * (acc_av + acc_va)

        else: # eval mode: we do not compute all-gather contrastive and use original setting
            audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
            video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
            if len(audio_rep.shape) == 1:
                audio_rep = audio_rep.unsqueeze(0)
                video_rep = video_rep.unsqueeze(0)
            total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05
            num_frames = total.shape[0] // (total.shape[1] // self.total_frame)
            video_indices = torch.arange(total.shape[1] // self.total_frame, device=total.device).repeat_interleave(self.total_frame)
            
            max_similarities, max_indices = torch.max(total, dim=1)
            correct_matches = (video_indices[max_indices] == torch.arange(total.shape[0] // self.total_frame, device=total.device).repeat_interleave(num_frames))
            
            c_acc = correct_matches.float().mean()
            
            # For NCE loss in eval mode, we can use the same formula but considering all frames from the same video as positive
            positive_mask = (video_indices.unsqueeze(0) == video_indices.unsqueeze(1))
            nce = -torch.mean(torch.logsumexp(total * positive_mask, dim=1) - torch.logsumexp(total, dim=1))

        return nce, c_acc

    # Make sure to update the forward_contrastive method in your CAVMAE class
    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False, mode='train'):
        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        if mode == 'train':
            if bidirect_contrast:
                nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
                nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
                c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
                c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
                nce = (nce_1 + nce_2) / 2
                c_acc = (c_acc_1 + c_acc_2) / 2
            else:
                nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
                c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
        else:  # eval mode
            # For eval, we consider any match within the same video as correct
            # This assumes that samples from the same video are grouped together
            num_frames = total.shape[0] // (total.shape[1] // self.total_frame)
            video_indices = torch.arange(total.shape[1] // self.total_frame, device=total.device).repeat_interleave(self.total_frame)
            
            max_similarities, max_indices = torch.max(total, dim=1)
            correct_matches = (video_indices[max_indices] == torch.arange(total.shape[0] // self.total_frame, device=total.device).repeat_interleave(num_frames))
            
            c_acc = correct_matches.float().mean()
            
            # For NCE loss in eval mode, we can use the same formula but considering all frames from the same video as positive
            positive_mask = (video_indices.unsqueeze(0) == video_indices.unsqueeze(1))
            nce = -torch.mean(torch.logsumexp(total * positive_mask, dim=1) - torch.logsumexp(total, dim=1))

        return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # for audio, need to adjust the shape
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 1, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), 16)
        elif modality == 'v':
            target = self.patchify(input, 3, int(input.shape[2]/self.patch_embed_v.patch_size[0]), int(input.shape[3]/self.patch_embed_v.patch_size[1]), 16)

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:

            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured', mode='train', 
                get_sample=False, distri=False, cls_mask_ratio=0.75, eval_mode=False, teacher_mode=False, rank_a=None, rank_v=None):
        
        if teacher_mode:
            mask_ratio_a, mask_ratio_v = 0.0, 0.0
            attn_a, attn_v, ca, cv = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode, teacher_mode=True)
            if self.probe_guide:
                rank_a, rank_v = self.get_patch2cls_rank(attn_a, mode='prob'), self.get_patch2cls_rank(attn_v, mode='prob')
            else:
                rank_a, rank_v = self.get_patch2cls_rank(attn_a, mode='deterministic'), self.get_patch2cls_rank(attn_v, mode='deterministic')
            return rank_a, rank_v, ca, cv
        
        if self.multi_ratio_masking:
            # Use same seed across all GPUs for each batch
            # Get same random number across all GPUs by using torch's random generator
            generator = torch.Generator(device=audio.device)
            # Set same seed across all GPUs
            if torch.distributed.is_initialized():
                # Ensure all GPUs use same seed by broadcasting from rank 0
                seed = torch.tensor(int(torch.empty((), device=audio.device).uniform_().item() * 1e6), device=audio.device)
                torch.distributed.broadcast(seed, src=0)
                generator.manual_seed(seed.item())
            else:
                generator.manual_seed(0)
            
            mask_ratio_a = 0.6 + 0.3 * torch.rand(1, device=audio.device, generator=generator).item()
            mask_ratio_v = 0.6 + 0.3 * torch.rand(1, device=audio.device, generator=generator).item()
        if self.cls_token:    
            latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v, cls_a, cls_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        else:
            latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        
        # MAE loss calculation (remains the same for both train and eval)
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)
            loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
            loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
            loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
        else:
            loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)


        # Contrastive loss calculation
        if self.cls_full and not eval_mode:
            mask_low = cls_mask_ratio
            if rank_a is not None and rank_v is not None:
                latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v, cls_a, cls_v = self.forward_encoder(audio, imgs, mask_low, mask_low, mask_mode=mask_mode, rank_a=rank_a, rank_v=rank_v)
            else:
                latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v, cls_a, cls_v = self.forward_encoder(audio, imgs, mask_low, mask_low, mask_mode=mask_mode, teacher_mode=False)

        if contrast_loss_weight != 0 and not distri:
            if not self.cls_token:
                loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1), mode=mode)
            else:
                if self.global_local_losses:
                    global_loss_c, global_c_acc = self.forward_contrastive(cls_a, cls_v, mode=mode)
                    local_loss_c, local_c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1), mode=mode)
                    loss_c = (global_loss_c + local_loss_c) / 2
                    c_acc = (local_c_acc + global_c_acc) / 2
                else:
                    loss_c, c_acc = self.forward_contrastive(cls_a, cls_v, mode=mode)
            loss_c = contrast_loss_weight * loss_c
        elif distri:
            loss_c, c_acc = self.forward_contrastive_all_gather(cls_a, cls_v, mode=mode, get_sample=get_sample)
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        loss = loss_mae + loss_c
        

        # reconstruct the audio and visual inputs for visualization
        recon_a = self.unpatchify(pred_a, 1, 8, int(self.audio_length/16), 16)
        recon_a = torch.einsum('nchw->nhwc', recon_a)
        recon_v = self.unpatchify(pred_v, 3, 14, 14, 16)
        recon_v = torch.einsum('nchw->nhwc', recon_v)

        if self.cls_token:
            if self.global_local_losses:
                return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, recon_a, recon_v, latent_c_a.mean(dim=1), latent_c_v.mean(dim=1), cls_a, cls_v, contrast_loss_weight * global_loss_c, contrast_loss_weight * local_loss_c
            # else:
            #     cls_a = latent_c_a
            #     cls_v = latent_c_v
        # else:
        #     cls_a = latent_c_a.mean(dim=1)
        #     cls_v = latent_c_v.mean(dim=1)

        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, recon_a, recon_v, cls_a, cls_v

    # used only for inpainting, ignore if inpainting is not of interest
    def forward_inpaint(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mask_mode='unstructured'):
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)  # [N, L, p*p*3]
        loss_pixel_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
        loss_pixel_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
        return pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v

    # used for retrieval, ignore if retrieval is not of interest
    def forward_feat(self, a, v, register_hook=False):
        if register_hook:
            self.register_hooks(self.blocks_a, "blocks_a_")
            self.register_hooks(self.blocks_v, "blocks_v_")
            self.register_hooks(self.blocks_u, "blocks_u_")
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v
		
        batch_size = a.shape[0]

        # Append cls tokens
        if self.cls_token:
            cls_tokens_a = self.cls_token_a.expand(batch_size, -1, -1)
            cls_tokens_v = self.cls_token_v.expand(batch_size, -1, -1)

            a = torch.cat([cls_tokens_a, a], dim=1)
            v = torch.cat([cls_tokens_v, v], dim=1)

        # Append register tokens
        if self.num_register_tokens > 0:
            
            r_a = self.register_tokens[:self.num_register_tokens].unsqueeze(0).expand(batch_size, -1, -1)
            r_v = self.register_tokens[self.num_register_tokens:].unsqueeze(0).expand(batch_size, -1, -1)
            
            a = torch.cat([a, r_a], dim=1)
            v = torch.cat([v, r_v], dim=1)

        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v:
            v = blk(v)

        if self.num_register_tokens > 0 and not self.keep_register_tokens:
            # Remove register tokens
            a = a[:, :-self.num_register_tokens, :]
            v = v[:, :-self.num_register_tokens, :]
        
        for blk in self.blocks_u:
            ca = blk(a, 'a')
        for blk in self.blocks_u:
            cv = blk(v, 'v')

        if self.contrastive_heads:
            for blk in self.constrative_head_audio:
                ca = blk(ca)
            for blk in self.constrative_head_visual:
                cv = blk(cv)
        
        if self.cls_token:
            # split the local patch tokens from the cls tokens
            #cls tokens
            cls_a = self.norm_a(ca[:, 0, :].squeeze())  # This gets the audio CLS token from ca
            cls_v = self.norm_v(cv[:, 0, :].squeeze())  # This gets the visual CLS token from cv
            # local patch tokens
            ca = self.norm_a(ca[:, 1:, :])
            cv = self.norm_v(cv[:, 1:, :])

            return ca, cv, cls_a, cls_v
        else:
            ca = self.norm_a(ca)
            cv = self.norm_v(cv)

        return ca, cv

# the finetuned CAV-MAE model
class CAVMAEFT(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True, aggregate='None', num_register_tokens=0, cls_token=False, total_frame=16, contrastive_head=False, joint_layers=1, keep_register_tokens=False, mode='multimodal'):
        super().__init__()
        timm.models.vision_transformer.Block = Block
        print('Use norm_pix_loss: ', norm_pix_loss)

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.keep_register_tokens = keep_register_tokens
        print('Keep Register Tokens: ', self.keep_register_tokens)
        self.aggregate = aggregate

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(modality_specific_depth)])
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(modality_specific_depth)])
        if joint_layers > 1:
            self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(joint_layers)])
        else:
            self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

        self.num_register_tokens = num_register_tokens
        print('Number of Registers: {:d}'.format(self.num_register_tokens))
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.num_register_tokens * 2, embed_dim))

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.cls_token = cls_token
        if self.cls_token:
            print("Using CLS Token")
            self.cls_token_a = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.cls_token_v = nn.Parameter(torch.randn(1, 1, embed_dim))


        self.mode = mode
        self.total_frame = total_frame
        print('Using {:d} frames'.format(self.total_frame))

        if self.aggregate == "concat_mlp":
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim * self.total_frame),
                nn.Linear(embed_dim * self.total_frame, embed_dim * 5),
                nn.GELU(),
                nn.Linear(embed_dim * (self.total_frame // 2), embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, label_dim)
            )
        elif self.aggregate == "self_attention_cls":
            if self.mode == 'multimodal':
                cls_dim = embed_dim*2
            else:
                cls_dim = embed_dim
            self.cls_cls_token = nn.Parameter(torch.zeros(1, 1, cls_dim))
            self.classifier_layers = nn.ModuleList([
                Block(cls_dim, num_heads, mlp_ratio, qkv_bias=True,  
                      norm_layer=norm_layer)
                for _ in range(2)  # 1. Add more dropout in the classifier layers
            ])
            self.classifier_norm = norm_layer(cls_dim)
            self.classifier_head = nn.Linear(cls_dim, label_dim)
            # # Add positional embedding for this transformer classifier
            # self.classifier_pos_embed = nn.Parameter(torch.zeros(1, total_frame+1, embed_dim), requires_grad=tr_pos)
        else:
            self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        print(test_output.shape)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        # classifier_seq_len = self.total_frame + 1
        # pos_embed_classifier = get_2d_sincos_pos_embed(self.classifier_pos_embed.shape[-1], int(classifier_seq_len ** .5), int(classifier_seq_len ** .5), cls_token=True)
        # self.classifier_pos_embed.data.copy_(torch.from_numpy(pos_embed_classifier).float().unsqueeze(0))
        
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.cls_token:
            torch.nn.init.normal_(self.cls_token_a, std=.02)
            torch.nn.init.normal_(self.cls_token_v, std=.02)

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        if self.aggregate == "self_attention_cls":
            torch.nn.init.normal_(self.cls_cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            batch_size = a.shape[0]

            #Append CLS tokens
            if self.cls_token:
                cls_tokens_a = self.cls_token_a.expand(batch_size, -1, -1)
                cls_tokens_v = self.cls_token_v.expand(batch_size, -1, -1)

                a = torch.cat([cls_tokens_a, a], dim=1)
                v = torch.cat([cls_tokens_v, v], dim=1)

            # Append register tokens
            if self.num_register_tokens > 0:
                batch_size = a.shape[0]
                r_a = self.register_tokens[:self.num_register_tokens].unsqueeze(0).expand(batch_size, -1, -1)
                r_v = self.register_tokens[self.num_register_tokens:].unsqueeze(0).expand(batch_size, -1, -1)
                
                a = torch.cat([a, r_a], dim=1)
                v = torch.cat([v, r_v], dim=1)

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            if self.num_register_tokens > 0 and not self.keep_register_tokens:
                # Remove register tokens
                a = a[:, :-self.num_register_tokens, :]
                v = v[:, :-self.num_register_tokens, :]

            num_a_tokens = a.shape[1]  # Includes CLS_A
            num_v_tokens = v.shape[1]  # Includes CLS_V

            # Concatenate audio and visual tokens with cls tokens
            x = torch.cat((a, v), dim=1) 

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            if self.num_register_tokens > 0 and self.keep_register_tokens:
                # Remove register tokens
                a = a[:, :-self.num_register_tokens, :]
                v = v[:, :-self.num_register_tokens, :]
                x = torch.cat((x[:, :self.patch_embed_a.num_patches, :], x[:, self.patch_embed_a.num_patches+self.num_register_tokens:-self.num_register_tokens, :]), dim=1)

            if self.cls_token:
                # Extract the cls tokens
                cls_tokens_a = a[:, 0, :]
                cls_tokens_v = v[:, 0, :]

                x = torch.cat((cls_tokens_a, cls_tokens_v), dim=1)
            else:
                a = a.mean(dim=1).squeeze()
                v = v.mean(dim=1).squeeze()
                x = torch.cat((a, v), dim=1)

            if self.aggregate == "self_attention_cls":
                # Reshape to (batch_size, no_frames_per_video, num_patches, embed_dim)
                batch_size = x.shape[0] // self.total_frame 
                x = x.view(batch_size, self.total_frame, x.shape[-1])
                
                cls_tokens = self.cls_cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                # x = x + classifier_pos_embed

                # Apply classifier layers
                for block in self.classifier_layers:
                    x = block(x)
                
                x = self.classifier_norm(x)
                x = self.classifier_head(x[:, 0])  # Use CLS token for classification
            elif self.aggregate != "None":
                # Reshape to (batch_size, no_frames_per_video, num_patches, embed_dim)
                batch_size = x.shape[0] // self.total_frame
                x = x.view(batch_size, self.total_frame, -1, x.shape[-1])
                
                # Average across patches
                x = x.mean(dim=2)
                
                # Concatenate frames
                # Expected dimension: (batch_size, self.total_frame * embed_dim)
                x = x.view(batch_size, -1)
                x = self.mlp_head(x)
            else:
                x = x.mean(dim=1)
                x = self.mlp_head(x)

            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            batch_size = a.shape[0]

            # Append CLS token if using
            if self.cls_token:
                cls_tokens_a = self.cls_token_a.expand(batch_size, -1, -1)
                a = torch.cat([cls_tokens_a, a], dim=1)

            # Append register tokens if using
            if self.num_register_tokens > 0:
                r_a = self.register_tokens[:self.num_register_tokens].unsqueeze(0).expand(batch_size, -1, -1)
                a = torch.cat([a, r_a], dim=1)

            # Process through audio blocks
            for blk in self.blocks_a:
                a = blk(a)

            # Remove register tokens if needed
            if self.num_register_tokens > 0 and not self.keep_register_tokens:
                a = a[:, :-self.num_register_tokens, :]

            # Process through unified blocks
            for blk in self.blocks_u:
                a = blk(a)
            a = self.norm(a)

            if self.aggregate == "self_attention_cls":
                # Reshape to (batch_size, no_frames_per_video, num_patches, embed_dim)
                batch_size = a.shape[0] // self.total_frame
                a = a.view(batch_size, self.total_frame, -1, a.shape[-1])
                
                # Average across patches
                a = a.mean(dim=2)
                
                # Add CLS token
                cls_tokens = self.cls_cls_token.expand(batch_size, -1, -1)
                a = torch.cat((cls_tokens, a), dim=1)
                
                # Apply classifier layers
                for block in self.classifier_layers:
                    a = block(a)
                
                a = self.classifier_norm(a)
                a = self.classifier_head(a[:, 0])  # Use CLS token for classification
                x = a
            else:
                if self.cls_token:
                    x = a[:, 0, :]  # Use CLS token
                else:
                    x = a.mean(dim=1)  # Average pooling
                x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only image when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            batch_size = v.shape[0]

            # Append CLS token if using
            if self.cls_token:
                cls_tokens_v = self.cls_token_v.expand(batch_size, -1, -1)
                v = torch.cat([cls_tokens_v, v], dim=1)

            # Append register tokens if using
            if self.num_register_tokens > 0:
                r_v = self.register_tokens[self.num_register_tokens:].unsqueeze(0).expand(batch_size, -1, -1)
                v = torch.cat([v, r_v], dim=1)

            # Process through visual blocks
            for blk in self.blocks_v:
                v = blk(v)

            # Remove register tokens if needed
            if self.num_register_tokens > 0 and not self.keep_register_tokens:
                v = v[:, :-self.num_register_tokens, :]

            # Process through unified blocks
            for blk in self.blocks_u:
                v = blk(v)
            v = self.norm(v)

            if self.aggregate == "self_attention_cls":
                # Reshape to (batch_size, no_frames_per_video, num_patches, embed_dim)
                batch_size = v.shape[0] // self.total_frame
                v = v.view(batch_size, self.total_frame, -1, v.shape[-1])

                # Average across patches
                v = v.mean(dim=2)
                
                # Add CLS token
                cls_tokens = self.cls_cls_token.expand(batch_size, -1, -1)
                v = torch.cat((cls_tokens, v), dim=1)
                
                # Apply classifier layers
                for block in self.classifier_layers:
                    v = block(v)
                
                v = self.classifier_norm(v)
                v = self.classifier_head(v[:, 0])  # Use CLS token for classification
                x = v
            else:
                if self.cls_token:
                    x = v[:, 0, :]  # Use CLS token
                else:
                    x = v.mean(dim=1)  # Average pooling
                x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode='av', register_hook=False):
        if register_hook:
            self.register_hooks(self.blocks_a, "blocks_a_")
            self.register_hooks(self.blocks_v, "blocks_v_")
            self.register_hooks(self.blocks_u, "blocks_u_")

        # return both audio and visual
        if mode == 'av':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return a, v

        # return only audio
        if mode == 'a':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            return a

    def get_features(self, a, v, mode='multimodal'):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            batch_size = a.shape[0]

            #Append CLS tokens
            if self.cls_token:
                cls_tokens_a = self.cls_token_a.expand(batch_size, -1, -1)
                cls_tokens_v = self.cls_token_v.expand(batch_size, -1, -1)

                a = torch.cat([cls_tokens_a, a], dim=1)
                v = torch.cat([cls_tokens_v, v], dim=1)

            # Append register tokens
            if self.num_register_tokens > 0:
                batch_size = a.shape[0]
                r_a = self.register_tokens[:self.num_register_tokens].unsqueeze(0).expand(batch_size, -1, -1)
                r_v = self.register_tokens[self.num_register_tokens:].unsqueeze(0).expand(batch_size, -1, -1)
                
                a = torch.cat([a, r_a], dim=1)
                v = torch.cat([v, r_v], dim=1)

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_v:
                v = blk(v)

            if self.num_register_tokens > 0 and not self.keep_register_tokens:
                # Remove register tokens
                a = a[:, :-self.num_register_tokens, :]
                v = v[:, :-self.num_register_tokens, :]

            num_a_tokens = a.shape[1]  # Includes CLS_A
            num_v_tokens = v.shape[1]  # Includes CLS_V

            # Concatenate audio and visual tokens with cls tokens
            x = torch.cat((a, v), dim=1) 

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            if self.num_register_tokens > 0 and self.keep_register_tokens:
                # Remove register tokens
                a = a[:, :-self.num_register_tokens, :]
                v = v[:, :-self.num_register_tokens, :]
                x = torch.cat((x[:, :self.patch_embed_a.num_patches, :], x[:, self.patch_embed_a.num_patches+self.num_register_tokens:-self.num_register_tokens, :]), dim=1)

            if self.cls_token:
                # Extract the cls tokens
                cls_tokens_a = a[:, 0, :]
                cls_tokens_v = v[:, 0, :]

                x = torch.cat((cls_tokens_a, cls_tokens_v), dim=1)
            else:
                a = a.mean(dim=1).squeeze()
                v = v.mean(dim=1).squeeze()
                x = torch.cat((a, v), dim=1)
            return x