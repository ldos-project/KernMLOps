/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * kermit.h - KernMLOps Insertion Interface
 *
 * Copyright (C) 2025 Aditya Tewari <adityaatewari@gmail.com>
 */

#ifndef _LINUX_KERMIT_H_
#define _LINUX_KERMIT_H_

#define ML_REPLACE_FUNCTION(type, mlfunc, mlinvoke, normal_expression) ({\
	type r; \
	if (mlfunc) r = (mlinvoke); \
	else r = ({normal_expression}); \
	r; \
})

#endif //_LINUX_KERMIT_H_
