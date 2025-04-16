	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function gemv_relu_kernel
.LCPI0_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
	.section	.rodata.cst8,"aM",@progbits,8
.LCPI0_1:
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	2                               # 0x2
	.byte	3                               # 0x3
	.byte	4                               # 0x4
	.byte	5                               # 0x5
	.byte	6                               # 0x6
	.byte	7                               # 0x7
	.text
	.globl	gemv_relu_kernel
	.p2align	4
	.type	gemv_relu_kernel,@function
gemv_relu_kernel:                       # @gemv_relu_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
                                        # kill: def $r8d killed $r8d def $r8
	movl	%ecx, -12(%rsp)                 # 4-byte Spill
	movq	%rdi, -8(%rsp)                  # 8-byte Spill
	movl	56(%rsp), %ecx
.Ltmp0:
	shll	$3, %ecx
.Ltmp1:
	leal	7(%r8), %r10d
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
.Ltmp2:
	cmpl	$8, %r10d
	jl	.LBB0_3
# %bb.1:                                # %.lr.ph
	movl	%ecx, %r11d
	imull	%r9d, %r11d
	andl	$-8, %r10d
	leal	7(%rcx), %ebx
	imull	%r9d, %ebx
	leal	6(%rcx), %r14d
	imull	%r9d, %r14d
	leal	5(%rcx), %r15d
	imull	%r9d, %r15d
	leal	4(%rcx), %r12d
	imull	%r9d, %r12d
	leal	3(%rcx), %r13d
	imull	%r9d, %r13d
	leal	2(%rcx), %ebp
	imull	%r9d, %ebp
	leal	1(%rcx), %eax
	imull	%r9d, %eax
	xorl	%r9d, %r9d
	vpmovsxbd	.LCPI0_1(%rip), %ymm2   # ymm2 = [0,1,2,3,4,5,6,7]
.Ltmp3:
	.p2align	4
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	vmovd	%r8d, %xmm3
	vpbroadcastd	%xmm3, %ymm3
	vpcmpgtd	%ymm2, %ymm3, %ymm3
	leal	(%r11,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm4
	leal	(%rax,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm5
	leal	(%rbp,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm6
	leal	(%r13,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm7
	leal	(%r12,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm8
	leal	(%r15,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm9
	leal	(%r14,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm10
	leal	(%rbx,%r9), %edi
	movslq	%edi, %rdi
	vmaskmovps	(%rsi,%rdi,4), %ymm3, %ymm11
	vmaskmovps	(%rdx,%r9,4), %ymm3, %ymm3
	vmulps	%ymm3, %ymm4, %ymm12
	vmulps	%ymm3, %ymm5, %ymm13
	vmulps	%ymm3, %ymm6, %ymm14
	vmulps	%ymm3, %ymm7, %ymm7
	vmulps	%ymm3, %ymm8, %ymm6
	vmulps	%ymm3, %ymm9, %ymm5
	vmulps	%ymm3, %ymm10, %ymm4
	vmulps	%ymm3, %ymm11, %ymm3
.Ltmp4:
	vextractf128	$1, %ymm12, %xmm8
	vaddps	%xmm8, %xmm12, %xmm8
	vshufpd	$1, %xmm8, %xmm8, %xmm9         # xmm9 = xmm8[1,0]
	vextractf128	$1, %ymm13, %xmm10
	vaddps	%xmm10, %xmm13, %xmm10
	vshufpd	$1, %xmm10, %xmm10, %xmm11      # xmm11 = xmm10[1,0]
	vextractf128	$1, %ymm14, %xmm12
	vaddps	%xmm12, %xmm14, %xmm12
	vshufpd	$1, %xmm12, %xmm12, %xmm13      # xmm13 = xmm12[1,0]
	vextractf128	$1, %ymm7, %xmm14
	vaddps	%xmm7, %xmm14, %xmm7
	vshufpd	$1, %xmm7, %xmm7, %xmm14        # xmm14 = xmm7[1,0]
	vaddps	%xmm11, %xmm10, %xmm10
	vaddps	%xmm9, %xmm8, %xmm8
	vunpcklps	%xmm10, %xmm8, %xmm9    # xmm9 = xmm8[0],xmm10[0],xmm8[1],xmm10[1]
	vaddps	%xmm13, %xmm12, %xmm11
	vmovlhps	%xmm11, %xmm9, %xmm9            # xmm9 = xmm9[0],xmm11[0]
	vaddps	%xmm7, %xmm14, %xmm7
	vinsertps	$48, %xmm7, %xmm9, %xmm9 # xmm9 = xmm9[0,1,2],xmm7[0]
	vinsertps	$76, %xmm8, %xmm10, %xmm8 # xmm8 = xmm8[1],xmm10[1],zero,zero
	vshufps	$212, %xmm11, %xmm8, %xmm8      # xmm8 = xmm8[0,1],xmm11[1,3]
	vinsertps	$112, %xmm7, %xmm8, %xmm7 # xmm7 = xmm8[0,1,2],xmm7[1]
	vextractf128	$1, %ymm6, %xmm8
	vaddps	%xmm6, %xmm8, %xmm6
	vshufpd	$9, %xmm6, %xmm6, %xmm8         # xmm8 = xmm6[1,0]
	vaddps	%xmm6, %xmm8, %xmm6
	vinsertf128	$1, %xmm6, %ymm9, %ymm8
	vmovshdup	%xmm6, %xmm6            # xmm6 = xmm6[1,1,3,3]
	vinsertf128	$1, %xmm6, %ymm0, %ymm6
	vblendps	$240, %ymm6, %ymm7, %ymm6       # ymm6 = ymm7[0,1,2,3],ymm6[4,5,6,7]
	vpermpd	$78, %ymm5, %ymm9               # ymm9 = ymm5[2,3,0,1]
	vaddps	%ymm5, %ymm9, %ymm5
	vshufpd	$9, %xmm5, %xmm5, %xmm9         # xmm9 = xmm5[1,0]
	vaddps	%ymm5, %ymm9, %ymm5
	vbroadcastss	%xmm5, %ymm9
	vblendps	$32, %ymm9, %ymm8, %ymm8        # ymm8 = ymm8[0,1,2,3,4],ymm9[5],ymm8[6,7]
	vinsertf128	$1, %xmm5, %ymm7, %ymm5
	vblendps	$34, %ymm5, %ymm6, %ymm5        # ymm5 = ymm6[0],ymm5[1],ymm6[2,3,4],ymm5[5],ymm6[6,7]
	vextractf128	$1, %ymm4, %xmm6
	vaddps	%xmm6, %xmm4, %xmm4
	vshufpd	$9, %xmm4, %xmm4, %xmm6         # xmm6 = xmm4[1,0]
	vaddps	%xmm6, %xmm4, %xmm4
	vbroadcastsd	%xmm4, %ymm6
	vblendps	$192, %ymm6, %ymm8, %ymm6       # ymm6 = ymm8[0,1,2,3,4,5],ymm6[6,7]
	vshufps	$85, %xmm4, %xmm4, %xmm4        # xmm4 = xmm4[1,1,1,1]
	vinsertf128	$1, %xmm4, %ymm0, %ymm4
	vblendps	$192, %ymm4, %ymm5, %ymm4       # ymm4 = ymm5[0,1,2,3,4,5],ymm4[6,7]
	vpermpd	$78, %ymm3, %ymm5               # ymm5 = ymm3[2,3,0,1]
	vaddps	%ymm5, %ymm3, %ymm3
	vshufpd	$9, %xmm3, %xmm3, %xmm5         # xmm5 = xmm3[1,0]
	vaddps	%ymm5, %ymm3, %ymm3
	vbroadcastss	%xmm3, %ymm5
	vblendps	$128, %ymm5, %ymm6, %ymm5       # ymm5 = ymm6[0,1,2,3,4,5,6],ymm5[7]
	vbroadcastsd	%xmm3, %ymm3
	vblendps	$128, %ymm3, %ymm4, %ymm3       # ymm3 = ymm4[0,1,2,3,4,5,6],ymm3[7]
	vaddps	%ymm3, %ymm5, %ymm3
.Ltmp5:
	vaddps	%ymm3, %ymm1, %ymm1
	addq	$8, %r9
	addl	$-8, %r8d
	cmpq	%r9, %r10
	jne	.LBB0_2
.LBB0_3:                                # %._crit_edge
	vmovd	%ecx, %xmm2
	vpbroadcastd	%xmm2, %ymm2
	vpor	.LCPI0_0(%rip), %ymm2, %ymm2
	vmaxps	%ymm0, %ymm1, %ymm0
	vmovd	-12(%rsp), %xmm1                # 4-byte Folded Reload
                                        # xmm1 = mem[0],zero,zero,zero
	vpbroadcastd	%xmm1, %ymm1
	vpcmpgtd	%ymm2, %ymm1, %ymm1
	movslq	%ecx, %rax
	movq	-8(%rsp), %rcx                  # 8-byte Reload
	vmaskmovps	%ymm0, %ymm1, (%rcx,%rax,4)
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	vzeroupper
	retq
.Ltmp6:
.Lfunc_end0:
	.size	gemv_relu_kernel, .Lfunc_end0-gemv_relu_kernel
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
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
	.byte	1                               # Abbrev [1] 0xb:0x60 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x30:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	42                              # DW_AT_abstract_origin
	.byte	4                               # Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              # DW_AT_abstract_origin
	.quad	.Ltmp1                          # DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	35                              # DW_AT_call_line
	.byte	27                              # DW_AT_call_column
	.byte	4                               # Abbrev [4] 0x55:0x14 DW_TAG_inlined_subroutine
	.long	42                              # DW_AT_abstract_origin
	.quad	.Ltmp4                          # DW_AT_low_pc
	.long	.Ltmp5-.Ltmp4                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	52                              # DW_AT_call_line
	.byte	22                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        # string offset=0
.Linfo_string1:
	.asciz	"gemv_kernel.py"                # string offset=7
.Linfo_string2:
	.asciz	"/users/uzairn/KernMLOps/Kdeploy/experiment/triton_kernels" # string offset=22
.Linfo_string3:
	.asciz	"gemv_relu_kernel"              # string offset=80
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
