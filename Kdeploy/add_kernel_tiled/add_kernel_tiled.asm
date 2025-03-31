	.text
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function add_kernel_tiled
.LCPI0_0:
	.long	8                               # 0x8
	.long	9                               # 0x9
	.long	10                              # 0xa
	.long	11                              # 0xb
	.long	12                              # 0xc
	.long	13                              # 0xd
	.long	14                              # 0xe
	.long	15                              # 0xf
.LCPI0_1:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
	.section	.rodata.cst8,"aM",@progbits,8
.LCPI0_2:
	.byte	8                               # 0x8
	.byte	9                               # 0x9
	.byte	10                              # 0xa
	.byte	11                              # 0xb
	.byte	12                              # 0xc
	.byte	13                              # 0xd
	.byte	14                              # 0xe
	.byte	15                              # 0xf
.LCPI0_3:
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	2                               # 0x2
	.byte	3                               # 0x3
	.byte	4                               # 0x4
	.byte	5                               # 0x5
	.byte	6                               # 0x6
	.byte	7                               # 0x7
	.text
	.globl	add_kernel_tiled
	.p2align	4
	.type	add_kernel_tiled,@function
add_kernel_tiled:                       # @add_kernel_tiled
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
                                        # kill: def $r8d killed $r8d def $r8
	shll	$12, %r8d
	vmovd	%ecx, %xmm0
	vpbroadcastd	%xmm0, %ymm0
	movslq	%r8d, %r9
	leaq	(%rdx,%r9,4), %rax
	leaq	(%rsi,%r9,4), %rcx
	leaq	(%rdi,%r9,4), %rdx
	xorl	%esi, %esi
	vpmovsxbd	.LCPI0_2(%rip), %ymm1   # ymm1 = [8,9,10,11,12,13,14,15]
	vpmovsxbd	.LCPI0_3(%rip), %ymm2   # ymm2 = [0,1,2,3,4,5,6,7]
.Ltmp0:
	.p2align	4
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	leal	(%r8,%rsi), %edi
	vmovd	%edi, %xmm3
	vpbroadcastd	%xmm3, %ymm3
	vpor	%ymm1, %ymm3, %ymm4
	vpor	%ymm2, %ymm3, %ymm3
	vpcmpgtd	%ymm3, %ymm0, %ymm3
	vpcmpgtd	%ymm4, %ymm0, %ymm4
	vmaskmovps	32(%rdx,%rsi,4), %ymm4, %ymm5
	vmaskmovps	(%rdx,%rsi,4), %ymm3, %ymm6
	vmaskmovps	32(%rcx,%rsi,4), %ymm4, %ymm7
	vaddps	%ymm7, %ymm5, %ymm5
	vmaskmovps	(%rcx,%rsi,4), %ymm3, %ymm7
	vmaskmovps	%ymm5, %ymm4, 32(%rax,%rsi,4)
	vaddps	%ymm7, %ymm6, %ymm4
	vmaskmovps	%ymm4, %ymm3, (%rax,%rsi,4)
	addq	$16, %rsi
	cmpq	$4096, %rsi                     # imm = 0x1000
	jne	.LBB0_1
# %bb.2:
	vzeroupper
	retq
.Ltmp1:
.Lfunc_end0:
	.size	add_kernel_tiled, .Lfunc_end0-add_kernel_tiled
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0
.Linfo_string1:
	.asciz	"add_kernel_tiled.py"           # string offset=7
.Linfo_string2:
	.asciz	"/home/uzairn/ldos-ml-compiler/kernel_modules/add_kernel_tiled" # string offset=27
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
