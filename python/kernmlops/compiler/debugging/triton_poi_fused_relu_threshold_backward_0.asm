    .file   "LLVMDialectModule"
    .section    .rodata.cst32,"aM",@progbits,32
    .p2align    5, 0x0                          # -- Begin function triton_poi_fused_relu_threshold_backward_0
.LCPI0_0:
    .long   32                              # 0x20
    .long   33                              # 0x21
    .long   34                              # 0x22
    .long   35                              # 0x23
    .long   36                              # 0x24
    .long   37                              # 0x25
    .long   38                              # 0x26
    .long   39                              # 0x27
.LCPI0_1:
    .long   40                              # 0x28
    .long   41                              # 0x29
    .long   42                              # 0x2a
    .long   43                              # 0x2b
    .long   44                              # 0x2c
    .long   45                              # 0x2d
    .long   46                              # 0x2e
    .long   47                              # 0x2f
.LCPI0_2:
    .long   48                              # 0x30
    .long   49                              # 0x31
    .long   50                              # 0x32
    .long   51                              # 0x33
    .long   52                              # 0x34
    .long   53                              # 0x35
    .long   54                              # 0x36
    .long   55                              # 0x37
.LCPI0_3:
    .long   56                              # 0x38
    .long   57                              # 0x39
    .long   58                              # 0x3a
    .long   59                              # 0x3b
    .long   60                              # 0x3c
    .long   61                              # 0x3d
    .long   62                              # 0x3e
    .long   63                              # 0x3f
.LCPI0_4:
    .long   0                               # 0x0
    .long   1                               # 0x1
    .long   2                               # 0x2
    .long   3                               # 0x3
    .long   4                               # 0x4
    .long   5                               # 0x5
    .long   6                               # 0x6
    .long   7                               # 0x7
.LCPI0_5:
    .long   8                               # 0x8
    .long   9                               # 0x9
    .long   10                              # 0xa
    .long   11                              # 0xb
    .long   12                              # 0xc
    .long   13                              # 0xd
    .long   14                              # 0xe
    .long   15                              # 0xf
.LCPI0_6:
    .long   16                              # 0x10
    .long   17                              # 0x11
    .long   18                              # 0x12
    .long   19                              # 0x13
    .long   20                              # 0x14
    .long   21                              # 0x15
    .long   22                              # 0x16
    .long   23                              # 0x17
.LCPI0_7:
    .long   24                              # 0x18
    .long   25                              # 0x19
    .long   26                              # 0x1a
    .long   27                              # 0x1b
    .long   28                              # 0x1c
    .long   29                              # 0x1d
    .long   30                              # 0x1e
    .long   31                              # 0x1f
.LCPI0_9:
    .zero   32,1
    .section    .rodata.cst4,"aM",@progbits,4
    .p2align    2, 0x0
.LCPI0_8:
    .long   50                              # 0x32
    .section    .rodata,"a",@progbits
.LCPI0_10:
    .byte   1                               # 0x1
    .text
    .globl  triton_poi_fused_relu_threshold_backward_0
    .p2align    4
    .type   triton_poi_fused_relu_threshold_backward_0,@function
triton_poi_fused_relu_threshold_backward_0: # @triton_poi_fused_relu_threshold_backward_0
.Lfunc_begin0:
    .file   1 "dump/5v" "c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py"
    .loc    1 18 0                          # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:18:0
    .cfi_sections .debug_frame
    .cfi_startproc
# %bb.0:
    subq    $104, %rsp
    .cfi_def_cfa_offset 112
.Ltmp0:
    .loc    1 21 23 prologue_end            # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:21:23
    vmovd   %ecx, %xmm0
    vpbroadcastd    %xmm0, %ymm0
    vpslld  $6, %ymm0, %ymm0
    vpor    .LCPI0_0(%rip), %ymm0, %ymm1
    vpor    .LCPI0_1(%rip), %ymm0, %ymm3
    vpor    .LCPI0_2(%rip), %ymm0, %ymm4
    vpor    .LCPI0_3(%rip), %ymm0, %ymm5
    vpor    .LCPI0_4(%rip), %ymm0, %ymm2
    vpor    .LCPI0_5(%rip), %ymm0, %ymm7
    vpor    .LCPI0_6(%rip), %ymm0, %ymm6
    vpor    .LCPI0_7(%rip), %ymm0, %ymm0
    .loc    1 20 33                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:20:33
    shll    $6, %ecx
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vpbroadcastd    .LCPI0_8(%rip), %ymm13  # ymm13 = [50,50,50,50,50,50,50,50]
    vpcmpgtd    %ymm0, %ymm13, %ymm8
    vpcmpgtd    %ymm6, %ymm13, %ymm6
    vpackssdw   %ymm8, %ymm6, %ymm0
    vpermq  $216, %ymm0, %ymm0              # ymm0 = ymm0[0,2,1,3]
    vmovdqu %ymm0, -64(%rsp)                # 32-byte Spill
    vpcmpgtd    %ymm7, %ymm13, %ymm7
    vpcmpgtd    %ymm2, %ymm13, %ymm15
    vpackssdw   %ymm7, %ymm15, %ymm2
    vpcmpgtd    %ymm5, %ymm13, %ymm10
    vpcmpgtd    %ymm4, %ymm13, %ymm11
    vpcmpgtd    %ymm3, %ymm13, %ymm12
    vpcmpgtd    %ymm1, %ymm13, %ymm13
    .loc    1 24 34                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:24:34
    movslq  %ecx, %rcx
    .loc    1 24 39 is_stmt 0               # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:24:39
    vmaskmovps  224(%rdi,%rcx,4), %ymm10, %ymm5
    vmaskmovps  192(%rdi,%rcx,4), %ymm11, %ymm4
    vmaskmovps  160(%rdi,%rcx,4), %ymm12, %ymm1
    vmaskmovps  128(%rdi,%rcx,4), %ymm13, %ymm3
    vmaskmovps  96(%rdi,%rcx,4), %ymm8, %ymm0
    vmovups %ymm0, -128(%rsp)               # 32-byte Spill
    .loc    1 22 21 is_stmt 1               # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vpermq  $216, %ymm2, %ymm9              # ymm9 = ymm2[0,2,1,3]
    .loc    1 24 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:24:39
    vxorps  %xmm0, %xmm0, %xmm0
.Ltmp1:
    .file   2 "/home/uzairn/github/KernMLOps/.venv/lib/python3.13/site-packages/torch/_inductor/runtime" "triton_helpers.py"
    .loc    2 114 29                        # triton_helpers.py:114:29 @[ c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:26:40 ]
    vmaxps  %ymm3, %ymm0, %ymm3
    vmaxps  %ymm1, %ymm0, %ymm2
    vmaxps  %ymm4, %ymm0, %ymm1
    vmaxps  %ymm5, %ymm0, %ymm0
.Ltmp2:
    .loc    1 24 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:24:39
    vmaskmovps  64(%rdi,%rcx,4), %ymm6, %ymm5
    vmaskmovps  32(%rdi,%rcx,4), %ymm7, %ymm4
    vmaskmovps  (%rdi,%rcx,4), %ymm15, %ymm14
    vmovups %ymm14, -96(%rsp)               # 32-byte Spill
    vmovups %ymm0, -32(%rsp)                # 32-byte Spill
    .loc    1 29 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:29:39
    vmaskmovps  %ymm0, %ymm10, 224(%rdi,%rcx,4)
    vmovups %ymm1, (%rsp)                   # 32-byte Spill
    vmaskmovps  %ymm1, %ymm11, 192(%rdi,%rcx,4)
    vmovups %ymm2, 32(%rsp)                 # 32-byte Spill
    vmaskmovps  %ymm2, %ymm12, 160(%rdi,%rcx,4)
    vmovups %ymm3, 64(%rsp)                 # 32-byte Spill
    vmaskmovps  %ymm3, %ymm13, 128(%rdi,%rcx,4)
    .loc    1 24 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:24:39
    vxorps  %xmm0, %xmm0, %xmm0
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vpacksswb   -64(%rsp), %ymm9, %ymm9 # 32-byte Folded Reload
.Ltmp3:
    .loc    2 114 29                        # triton_helpers.py:114:29 @[ c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:26:40 ]
    vmaxps  -128(%rsp), %ymm0, %ymm14       # 32-byte Folded Reload
.Ltmp4:
    .loc    1 29 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:29:39
    vmaskmovps  %ymm14, %ymm8, 96(%rdi,%rcx,4)
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vxorps  %xmm8, %xmm8, %xmm8
    vpermq  $216, %ymm9, %ymm8              # ymm8 = ymm9[0,2,1,3]
    vpmovmskb   %ymm8, %edx
    vpackssdw   %ymm10, %ymm11, %ymm8
.Ltmp5:
    .loc    2 114 29                        # triton_helpers.py:114:29 @[ c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:26:40 ]
    vmaxps  %ymm5, %ymm0, %ymm1
.Ltmp6:
    .loc    1 29 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:29:39
    vmaskmovps  %ymm1, %ymm6, 64(%rdi,%rcx,4)
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vpackssdw   %ymm12, %ymm13, %ymm6
.Ltmp7:
    .loc    2 114 29                        # triton_helpers.py:114:29 @[ c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:26:40 ]
    vmaxps  %ymm4, %ymm0, %ymm3
.Ltmp8:
    .loc    1 29 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:29:39
    vmaskmovps  %ymm3, %ymm7, 32(%rdi,%rcx,4)
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vxorps  %xmm7, %xmm7, %xmm7
    vpermq  $216, %ymm8, %ymm7              # ymm7 = ymm8[0,2,1,3]
    vpermq  $216, %ymm6, %ymm6              # ymm6 = ymm6[0,2,1,3]
    vpacksswb   %ymm7, %ymm6, %ymm6
.Ltmp9:
    .loc    2 114 29                        # triton_helpers.py:114:29 @[ c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:26:40 ]
    vmaxps  -96(%rsp), %ymm0, %ymm4         # 32-byte Folded Reload
.Ltmp10:
    .loc    1 29 39                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:29:39
    vmaskmovps  %ymm4, %ymm15, (%rdi,%rcx,4)
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    vxorps  %xmm2, %xmm2, %xmm2
    vpermq  $216, %ymm6, %ymm2              # ymm2 = ymm6[0,2,1,3]
    vpmovmskb   %ymm2, %eax
    .loc    1 30 36                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:36
    vcmpleps    %ymm0, %ymm14, %ymm2
    vcmpleps    %ymm0, %ymm1, %ymm1
    vpackssdw   %ymm2, %ymm1, %ymm1
    vcmpleps    %ymm0, %ymm3, %ymm2
    vcmpleps    %ymm0, %ymm4, %ymm3
    vpackssdw   %ymm2, %ymm3, %ymm2
    .loc    1 22 21                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:22:21
    shlq    $32, %rax
    orq %rdx, %rax
    .loc    1 30 36                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:36
    vpermq  $216, %ymm1, %ymm1              # ymm1 = ymm1[0,2,1,3]
    vpermq  $216, %ymm2, %ymm2              # ymm2 = ymm2[0,2,1,3]
    vpacksswb   %ymm1, %ymm2, %ymm1
    vpbroadcastb    .LCPI0_10(%rip), %ymm2  # ymm2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    vpermq  $216, %ymm1, %ymm1              # ymm1 = ymm1[0,2,1,3]
    vpand   %ymm2, %ymm1, %ymm6
    addq    %rcx, %rsi
    testb   $1, %al
    jne .LBB0_1
# %bb.2:                                # %else
    testb   $2, %al
    jne .LBB0_3
.LBB0_4:                                # %else2
    testb   $4, %al
    jne .LBB0_5
.LBB0_6:                                # %else4
    testb   $8, %al
    jne .LBB0_7
.LBB0_8:                                # %else6
    testb   $16, %al
    jne .LBB0_9
.LBB0_10:                               # %else8
    testb   $32, %al
    jne .LBB0_11
.LBB0_12:                               # %else10
    testb   $64, %al
    jne .LBB0_13
.LBB0_14:                               # %else12
    testb   %al, %al
    js  .LBB0_15
.LBB0_16:                               # %else14
    testl   $256, %eax                      # imm = 0x100
    jne .LBB0_17
.LBB0_18:                               # %else16
    testl   $512, %eax                      # imm = 0x200
    jne .LBB0_19
.LBB0_20:                               # %else18
    testl   $1024, %eax                     # imm = 0x400
    jne .LBB0_21
.LBB0_22:                               # %else20
    testl   $2048, %eax                     # imm = 0x800
    jne .LBB0_23
.LBB0_24:                               # %else22
    testl   $4096, %eax                     # imm = 0x1000
    jne .LBB0_25
.LBB0_26:                               # %else24
    testl   $8192, %eax                     # imm = 0x2000
    jne .LBB0_27
.LBB0_28:                               # %else26
    testl   $16384, %eax                    # imm = 0x4000
    jne .LBB0_29
.LBB0_30:                               # %else28
    testw   %ax, %ax
    js  .LBB0_31
.LBB0_32:                               # %else30
    .loc    1 0 36 is_stmt 0                # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:0:36
    vextracti128    $1, %ymm6, %xmm6
    .loc    1 30 36                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:36
    testl   $65536, %eax                    # imm = 0x10000
    jne .LBB0_33
.LBB0_34:                               # %else32
    testl   $131072, %eax                   # imm = 0x20000
    jne .LBB0_35
.LBB0_36:                               # %else34
    testl   $262144, %eax                   # imm = 0x40000
    jne .LBB0_37
.LBB0_38:                               # %else36
    testl   $524288, %eax                   # imm = 0x80000
    jne .LBB0_39
.LBB0_40:                               # %else38
    testl   $1048576, %eax                  # imm = 0x100000
    jne .LBB0_41
.LBB0_42:                               # %else40
    testl   $2097152, %eax                  # imm = 0x200000
    jne .LBB0_43
.LBB0_44:                               # %else42
    testl   $4194304, %eax                  # imm = 0x400000
    jne .LBB0_45
.LBB0_46:                               # %else44
    testl   $8388608, %eax                  # imm = 0x800000
    jne .LBB0_47
.LBB0_48:                               # %else46
    testl   $16777216, %eax                 # imm = 0x1000000
    jne .LBB0_49
.LBB0_50:                               # %else48
    testl   $33554432, %eax                 # imm = 0x2000000
    jne .LBB0_51
.LBB0_52:                               # %else50
    testl   $67108864, %eax                 # imm = 0x4000000
    je  .LBB0_54
.LBB0_53:                               # %cond.store51
    vpextrb $10, %xmm6, 26(%rsi)
.LBB0_54:                               # %else52
    .loc    1 0 36                          # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:0:36
    vmovups -32(%rsp), %ymm1                # 32-byte Reload
    .loc    1 30 36                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:36
    vcmpleps    %ymm0, %ymm1, %ymm5
    vmovups (%rsp), %ymm1                   # 32-byte Reload
    vcmpleps    %ymm0, %ymm1, %ymm4
    vmovups 32(%rsp), %ymm1                 # 32-byte Reload
    vcmpleps    %ymm0, %ymm1, %ymm3
    vmovups 64(%rsp), %ymm1                 # 32-byte Reload
    vcmpleps    %ymm0, %ymm1, %ymm1
    testl   $134217728, %eax                # imm = 0x8000000
    jne .LBB0_55
# %bb.56:                               # %else54
    vpackssdw   %ymm5, %ymm4, %ymm0
    vpackssdw   %ymm3, %ymm1, %ymm1
    testl   $268435456, %eax                # imm = 0x10000000
    jne .LBB0_57
.LBB0_58:                               # %else56
    vpermq  $216, %ymm0, %ymm0              # ymm0 = ymm0[0,2,1,3]
    vpermq  $216, %ymm1, %ymm1              # ymm1 = ymm1[0,2,1,3]
    testl   $536870912, %eax                # imm = 0x20000000
    jne .LBB0_59
.LBB0_60:                               # %else58
    vpacksswb   %ymm0, %ymm1, %ymm0
    testl   $1073741824, %eax               # imm = 0x40000000
    jne .LBB0_61
.LBB0_62:                               # %else60
    vpermq  $216, %ymm0, %ymm0              # ymm0 = ymm0[0,2,1,3]
    testl   %eax, %eax
    js  .LBB0_63
.LBB0_64:                               # %else62
    vpand   %ymm2, %ymm0, %ymm0
    btq $32, %rax
    jb  .LBB0_65
.LBB0_66:                               # %else64
    btq $33, %rax
    jb  .LBB0_67
.LBB0_68:                               # %else66
    btq $34, %rax
    jb  .LBB0_69
.LBB0_70:                               # %else68
    btq $35, %rax
    jb  .LBB0_71
.LBB0_72:                               # %else70
    btq $36, %rax
    jb  .LBB0_73
.LBB0_74:                               # %else72
    btq $37, %rax
    jb  .LBB0_75
.LBB0_76:                               # %else74
    btq $38, %rax
    jb  .LBB0_77
.LBB0_78:                               # %else76
    btq $39, %rax
    jb  .LBB0_79
.LBB0_80:                               # %else78
    btq $40, %rax
    jb  .LBB0_81
.LBB0_82:                               # %else80
    btq $41, %rax
    jb  .LBB0_83
.LBB0_84:                               # %else82
    btq $42, %rax
    jb  .LBB0_85
.LBB0_86:                               # %else84
    btq $43, %rax
    jb  .LBB0_87
.LBB0_88:                               # %else86
    btq $44, %rax
    jb  .LBB0_89
.LBB0_90:                               # %else88
    btq $45, %rax
    jb  .LBB0_91
.LBB0_92:                               # %else90
    btq $46, %rax
    jb  .LBB0_93
.LBB0_94:                               # %else92
    btq $47, %rax
    jb  .LBB0_95
.LBB0_96:                               # %else94
    btq $48, %rax
    vextracti128    $1, %ymm0, %xmm0
    jb  .LBB0_97
.LBB0_98:                               # %else96
    btq $49, %rax
    jb  .LBB0_99
.LBB0_100:                              # %else98
    btq $50, %rax
    jb  .LBB0_101
.LBB0_102:                              # %else100
    btq $51, %rax
    jb  .LBB0_103
.LBB0_104:                              # %else102
    btq $52, %rax
    jb  .LBB0_105
.LBB0_106:                              # %else104
    btq $53, %rax
    jb  .LBB0_107
.LBB0_108:                              # %else106
    btq $54, %rax
    jb  .LBB0_109
.LBB0_110:                              # %else108
    btq $55, %rax
    jb  .LBB0_111
.LBB0_112:                              # %else110
    btq $56, %rax
    jb  .LBB0_113
.LBB0_114:                              # %else112
    btq $57, %rax
    jb  .LBB0_115
.LBB0_116:                              # %else114
    btq $58, %rax
    jb  .LBB0_117
.LBB0_118:                              # %else116
    btq $59, %rax
    jb  .LBB0_119
.LBB0_120:                              # %else118
    btq $60, %rax
    jb  .LBB0_121
.LBB0_122:                              # %else120
    btq $61, %rax
    jb  .LBB0_123
.LBB0_124:                              # %else122
    btq $62, %rax
    jb  .LBB0_125
.LBB0_126:                              # %else124
    btq $63, %rax
    jb  .LBB0_127
.LBB0_128:                              # %else126
    .loc    1 30 4 epilogue_begin           # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:4
    addq    $104, %rsp
    .cfi_def_cfa_offset 8
    vzeroupper
    retq
.LBB0_1:                                # %cond.store
    .cfi_def_cfa_offset 112
    .loc    1 30 36                         # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:36
    vpextrb $0, %xmm6, (%rsi)
    testb   $2, %al
    je  .LBB0_4
.LBB0_3:                                # %cond.store1
    vpextrb $1, %xmm6, 1(%rsi)
    testb   $4, %al
    je  .LBB0_6
.LBB0_5:                                # %cond.store3
    vpextrb $2, %xmm6, 2(%rsi)
    testb   $8, %al
    je  .LBB0_8
.LBB0_7:                                # %cond.store5
    vpextrb $3, %xmm6, 3(%rsi)
    testb   $16, %al
    je  .LBB0_10
.LBB0_9:                                # %cond.store7
    vpextrb $4, %xmm6, 4(%rsi)
    testb   $32, %al
    je  .LBB0_12
.LBB0_11:                               # %cond.store9
    vpextrb $5, %xmm6, 5(%rsi)
    testb   $64, %al
    je  .LBB0_14
.LBB0_13:                               # %cond.store11
    vpextrb $6, %xmm6, 6(%rsi)
    testb   %al, %al
    jns .LBB0_16
.LBB0_15:                               # %cond.store13
    vpextrb $7, %xmm6, 7(%rsi)
    testl   $256, %eax                      # imm = 0x100
    je  .LBB0_18
.LBB0_17:                               # %cond.store15
    vpextrb $8, %xmm6, 8(%rsi)
    testl   $512, %eax                      # imm = 0x200
    je  .LBB0_20
.LBB0_19:                               # %cond.store17
    vpextrb $9, %xmm6, 9(%rsi)
    testl   $1024, %eax                     # imm = 0x400
    je  .LBB0_22
.LBB0_21:                               # %cond.store19
    vpextrb $10, %xmm6, 10(%rsi)
    testl   $2048, %eax                     # imm = 0x800
    je  .LBB0_24
.LBB0_23:                               # %cond.store21
    vpextrb $11, %xmm6, 11(%rsi)
    testl   $4096, %eax                     # imm = 0x1000
    je  .LBB0_26
.LBB0_25:                               # %cond.store23
    vpextrb $12, %xmm6, 12(%rsi)
    testl   $8192, %eax                     # imm = 0x2000
    je  .LBB0_28
.LBB0_27:                               # %cond.store25
    vpextrb $13, %xmm6, 13(%rsi)
    testl   $16384, %eax                    # imm = 0x4000
    je  .LBB0_30
.LBB0_29:                               # %cond.store27
    vpextrb $14, %xmm6, 14(%rsi)
    testw   %ax, %ax
    jns .LBB0_32
.LBB0_31:                               # %cond.store29
    vpextrb $15, %xmm6, 15(%rsi)
    vextracti128    $1, %ymm6, %xmm6
    testl   $65536, %eax                    # imm = 0x10000
    je  .LBB0_34
.LBB0_33:                               # %cond.store31
    vpextrb $0, %xmm6, 16(%rsi)
    testl   $131072, %eax                   # imm = 0x20000
    je  .LBB0_36
.LBB0_35:                               # %cond.store33
    vpextrb $1, %xmm6, 17(%rsi)
    testl   $262144, %eax                   # imm = 0x40000
    je  .LBB0_38
.LBB0_37:                               # %cond.store35
    vpextrb $2, %xmm6, 18(%rsi)
    testl   $524288, %eax                   # imm = 0x80000
    je  .LBB0_40
.LBB0_39:                               # %cond.store37
    vpextrb $3, %xmm6, 19(%rsi)
    testl   $1048576, %eax                  # imm = 0x100000
    je  .LBB0_42
.LBB0_41:                               # %cond.store39
    vpextrb $4, %xmm6, 20(%rsi)
    testl   $2097152, %eax                  # imm = 0x200000
    je  .LBB0_44
.LBB0_43:                               # %cond.store41
    vpextrb $5, %xmm6, 21(%rsi)
    testl   $4194304, %eax                  # imm = 0x400000
    je  .LBB0_46
.LBB0_45:                               # %cond.store43
    vpextrb $6, %xmm6, 22(%rsi)
    testl   $8388608, %eax                  # imm = 0x800000
    je  .LBB0_48
.LBB0_47:                               # %cond.store45
    vpextrb $7, %xmm6, 23(%rsi)
    testl   $16777216, %eax                 # imm = 0x1000000
    je  .LBB0_50
.LBB0_49:                               # %cond.store47
    vpextrb $8, %xmm6, 24(%rsi)
    testl   $33554432, %eax                 # imm = 0x2000000
    je  .LBB0_52
.LBB0_51:                               # %cond.store49
    vpextrb $9, %xmm6, 25(%rsi)
    testl   $67108864, %eax                 # imm = 0x4000000
    jne .LBB0_53
    jmp .LBB0_54
.LBB0_55:                               # %cond.store53
    vpextrb $11, %xmm6, 27(%rsi)
    vpackssdw   %ymm5, %ymm4, %ymm0
    vpackssdw   %ymm3, %ymm1, %ymm1
    testl   $268435456, %eax                # imm = 0x10000000
    je  .LBB0_58
.LBB0_57:                               # %cond.store55
    vpextrb $12, %xmm6, 28(%rsi)
    vpermq  $216, %ymm0, %ymm0              # ymm0 = ymm0[0,2,1,3]
    vpermq  $216, %ymm1, %ymm1              # ymm1 = ymm1[0,2,1,3]
    testl   $536870912, %eax                # imm = 0x20000000
    je  .LBB0_60
.LBB0_59:                               # %cond.store57
    vpextrb $13, %xmm6, 29(%rsi)
    vpacksswb   %ymm0, %ymm1, %ymm0
    testl   $1073741824, %eax               # imm = 0x40000000
    je  .LBB0_62
.LBB0_61:                               # %cond.store59
    vpextrb $14, %xmm6, 30(%rsi)
    vpermq  $216, %ymm0, %ymm0              # ymm0 = ymm0[0,2,1,3]
    testl   %eax, %eax
    jns .LBB0_64
.LBB0_63:                               # %cond.store61
    vpextrb $15, %xmm6, 31(%rsi)
    vpand   %ymm2, %ymm0, %ymm0
    btq $32, %rax
    jae .LBB0_66
.LBB0_65:                               # %cond.store63
    vpextrb $0, %xmm0, 32(%rsi)
    btq $33, %rax
    jae .LBB0_68
.LBB0_67:                               # %cond.store65
    vpextrb $1, %xmm0, 33(%rsi)
    btq $34, %rax
    jae .LBB0_70
.LBB0_69:                               # %cond.store67
    vpextrb $2, %xmm0, 34(%rsi)
    btq $35, %rax
    jae .LBB0_72
.LBB0_71:                               # %cond.store69
    vpextrb $3, %xmm0, 35(%rsi)
    btq $36, %rax
    jae .LBB0_74
.LBB0_73:                               # %cond.store71
    vpextrb $4, %xmm0, 36(%rsi)
    btq $37, %rax
    jae .LBB0_76
.LBB0_75:                               # %cond.store73
    vpextrb $5, %xmm0, 37(%rsi)
    btq $38, %rax
    jae .LBB0_78
.LBB0_77:                               # %cond.store75
    vpextrb $6, %xmm0, 38(%rsi)
    btq $39, %rax
    jae .LBB0_80
.LBB0_79:                               # %cond.store77
    vpextrb $7, %xmm0, 39(%rsi)
    btq $40, %rax
    jae .LBB0_82
.LBB0_81:                               # %cond.store79
    vpextrb $8, %xmm0, 40(%rsi)
    btq $41, %rax
    jae .LBB0_84
.LBB0_83:                               # %cond.store81
    vpextrb $9, %xmm0, 41(%rsi)
    btq $42, %rax
    jae .LBB0_86
.LBB0_85:                               # %cond.store83
    vpextrb $10, %xmm0, 42(%rsi)
    btq $43, %rax
    jae .LBB0_88
.LBB0_87:                               # %cond.store85
    vpextrb $11, %xmm0, 43(%rsi)
    btq $44, %rax
    jae .LBB0_90
.LBB0_89:                               # %cond.store87
    vpextrb $12, %xmm0, 44(%rsi)
    btq $45, %rax
    jae .LBB0_92
.LBB0_91:                               # %cond.store89
    vpextrb $13, %xmm0, 45(%rsi)
    btq $46, %rax
    jae .LBB0_94
.LBB0_93:                               # %cond.store91
    vpextrb $14, %xmm0, 46(%rsi)
    btq $47, %rax
    jae .LBB0_96
.LBB0_95:                               # %cond.store93
    vpextrb $15, %xmm0, 47(%rsi)
    btq $48, %rax
    vextracti128    $1, %ymm0, %xmm0
    jae .LBB0_98
.LBB0_97:                               # %cond.store95
    vpextrb $0, %xmm0, 48(%rsi)
    btq $49, %rax
    jae .LBB0_100
.LBB0_99:                               # %cond.store97
    vpextrb $1, %xmm0, 49(%rsi)
    btq $50, %rax
    jae .LBB0_102
.LBB0_101:                              # %cond.store99
    vpextrb $2, %xmm0, 50(%rsi)
    btq $51, %rax
    jae .LBB0_104
.LBB0_103:                              # %cond.store101
    vpextrb $3, %xmm0, 51(%rsi)
    btq $52, %rax
    jae .LBB0_106
.LBB0_105:                              # %cond.store103
    vpextrb $4, %xmm0, 52(%rsi)
    btq $53, %rax
    jae .LBB0_108
.LBB0_107:                              # %cond.store105
    vpextrb $5, %xmm0, 53(%rsi)
    btq $54, %rax
    jae .LBB0_110
.LBB0_109:                              # %cond.store107
    vpextrb $6, %xmm0, 54(%rsi)
    btq $55, %rax
    jae .LBB0_112
.LBB0_111:                              # %cond.store109
    vpextrb $7, %xmm0, 55(%rsi)
    btq $56, %rax
    jae .LBB0_114
.LBB0_113:                              # %cond.store111
    vpextrb $8, %xmm0, 56(%rsi)
    btq $57, %rax
    jae .LBB0_116
.LBB0_115:                              # %cond.store113
    vpextrb $9, %xmm0, 57(%rsi)
    btq $58, %rax
    jae .LBB0_118
.LBB0_117:                              # %cond.store115
    vpextrb $10, %xmm0, 58(%rsi)
    btq $59, %rax
    jae .LBB0_120
.LBB0_119:                              # %cond.store117
    vpextrb $11, %xmm0, 59(%rsi)
    btq $60, %rax
    jae .LBB0_122
.LBB0_121:                              # %cond.store119
    vpextrb $12, %xmm0, 60(%rsi)
    btq $61, %rax
    jae .LBB0_124
.LBB0_123:                              # %cond.store121
    vpextrb $13, %xmm0, 61(%rsi)
    btq $62, %rax
    jae .LBB0_126
.LBB0_125:                              # %cond.store123
    vpextrb $14, %xmm0, 62(%rsi)
    btq $63, %rax
    jae .LBB0_128
.LBB0_127:                              # %cond.store125
    vpextrb $15, %xmm0, 63(%rsi)
    .loc    1 30 4 epilogue_begin           # c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py:30:4
    addq    $104, %rsp
    .cfi_def_cfa_offset 8
    vzeroupper
    retq
.Ltmp11:
.Lfunc_end0:
    .size   triton_poi_fused_relu_threshold_backward_0, .Lfunc_end0-triton_poi_fused_relu_threshold_backward_0
    .cfi_endproc
                                        # -- End function
    .section    .debug_abbrev,"",@progbits
    .byte   1                               # Abbreviation Code
    .byte   17                              # DW_TAG_compile_unit
    .byte   1                               # DW_CHILDREN_yes
    .byte   37                              # DW_AT_producer
    .byte   14                              # DW_FORM_strp
    .byte   19                              # DW_AT_language
    .byte   5                               # DW_FORM_data2
    .byte   3                               # DW_AT_name
    .byte   14                              # DW_FORM_strp
    .byte   16                              # DW_AT_stmt_list
    .byte   23                              # DW_FORM_sec_offset
    .byte   27                              # DW_AT_comp_dir
    .byte   14                              # DW_FORM_strp
    .byte   17                              # DW_AT_low_pc
    .byte   1                               # DW_FORM_addr
    .byte   18                              # DW_AT_high_pc
    .byte   6                               # DW_FORM_data4
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
    .byte   2                               # Abbreviation Code
    .byte   46                              # DW_TAG_subprogram
    .byte   0                               # DW_CHILDREN_no
    .byte   3                               # DW_AT_name
    .byte   14                              # DW_FORM_strp
    .byte   32                              # DW_AT_inline
    .byte   11                              # DW_FORM_data1
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
    .byte   3                               # Abbreviation Code
    .byte   46                              # DW_TAG_subprogram
    .byte   1                               # DW_CHILDREN_yes
    .byte   17                              # DW_AT_low_pc
    .byte   1                               # DW_FORM_addr
    .byte   18                              # DW_AT_high_pc
    .byte   6                               # DW_FORM_data4
    .byte   49                              # DW_AT_abstract_origin
    .byte   19                              # DW_FORM_ref4
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
    .byte   4                               # Abbreviation Code
    .byte   29                              # DW_TAG_inlined_subroutine
    .byte   0                               # DW_CHILDREN_no
    .byte   49                              # DW_AT_abstract_origin
    .byte   19                              # DW_FORM_ref4
    .byte   85                              # DW_AT_ranges
    .byte   23                              # DW_FORM_sec_offset
    .byte   88                              # DW_AT_call_file
    .byte   11                              # DW_FORM_data1
    .byte   89                              # DW_AT_call_line
    .byte   11                              # DW_FORM_data1
    .byte   87                              # DW_AT_call_column
    .byte   11                              # DW_FORM_data1
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
    .byte   0                               # EOM(3)
    .section    .debug_info,"",@progbits
.Lcu_begin0:
    .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
    .short  4                               # DWARF version number
    .long   .debug_abbrev                   # Offset Into Abbrev. Section
    .byte   8                               # Address Size (in bytes)
    .byte   1                               # Abbrev [1] 0xb:0x44 DW_TAG_compile_unit
    .long   .Linfo_string0                  # DW_AT_producer
    .short  2                               # DW_AT_language
    .long   .Linfo_string1                  # DW_AT_name
    .long   .Lline_table_start0             # DW_AT_stmt_list
    .long   .Linfo_string2                  # DW_AT_comp_dir
    .quad   .Lfunc_begin0                   # DW_AT_low_pc
    .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
    .byte   2                               # Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
    .long   .Linfo_string3                  # DW_AT_name
    .byte   1                               # DW_AT_inline
    .byte   3                               # Abbrev [3] 0x30:0x1e DW_TAG_subprogram
    .quad   .Lfunc_begin0                   # DW_AT_low_pc
    .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
    .long   42                              # DW_AT_abstract_origin
    .byte   4                               # Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
    .long   42                              # DW_AT_abstract_origin
    .long   .Ldebug_ranges0                 # DW_AT_ranges
    .byte   1                               # DW_AT_call_file
    .byte   26                              # DW_AT_call_line
    .byte   40                              # DW_AT_call_column
    .byte   0                               # End Of Children Mark
    .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
    .section    .debug_ranges,"",@progbits
.Ldebug_ranges0:
    .quad   .Ltmp1-.Lfunc_begin0
    .quad   .Ltmp2-.Lfunc_begin0
    .quad   .Ltmp3-.Lfunc_begin0
    .quad   .Ltmp4-.Lfunc_begin0
    .quad   .Ltmp5-.Lfunc_begin0
    .quad   .Ltmp6-.Lfunc_begin0
    .quad   .Ltmp7-.Lfunc_begin0
    .quad   .Ltmp8-.Lfunc_begin0
    .quad   .Ltmp9-.Lfunc_begin0
    .quad   .Ltmp10-.Lfunc_begin0
    .quad   0
    .quad   0
    .section    .debug_str,"MS",@progbits,1
.Linfo_string0:
    .asciz  "triton"                        # string offset=0
.Linfo_string1:
    .asciz  "c5vok6xoydbh7civogkpvfuriablcfh4dso64uab7fb7vay2ffxf.py" # string offset=7
.Linfo_string2:
    .asciz  "dump/5v"                       # string offset=63
.Linfo_string3:
    .asciz  "triton_poi_fused_relu_threshold_backward_0" # string offset=71
    .section    ".note.GNU-stack","",@progbits
    .section    .debug_line,"",@progbits
.Lline_table_start0:
