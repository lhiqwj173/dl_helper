import torch

from dl_helper.tester import test_base
from dl_helper.train_param import Params

from dl_helper.data import data_parm2str
from dl_helper.models.binctabl import m_bin_ctabl
from dl_helper.transforms.binctabl import transform

"""
/usr/local/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
/usr/local/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
-------------------batch size: 1024 (0)-----------------------
/usr/local/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
E0710 04:40:01.023624401     451 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:40:01.023605435+00:00"}
E0710 04:40:01.023777623     445 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:40:01.023760232+00:00"}
E0710 04:40:01.023963488     449 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:40:01.023948336+00:00"}
E0710 04:40:01.031382849     447 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T04:40:01.031365862+00:00", grpc_status:2}
ddp: True
if_tqdm: False
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
learning_rate: 0.01664 -> 0.13312
batch_size: 1024
dataset length: 272955
dataloader length: 33
init train data done CPU 内存占用：3.4% (9.006GB/334.562GB)
broadcast_master_param
prepare done
each epoch step: 33
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank3]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank7]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank5]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank0]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[0][34] val done CPU 内存占用：12.5% (39.607GB/334.562GB)
[1][34] val done CPU 内存占用：13.8% (43.783GB/334.562GB)
[2][34] val done CPU 内存占用：14.3% (45.580GB/334.562GB)
[3][34] val done CPU 内存占用：14.9% (47.448GB/334.562GB)
[4][34] val done CPU 内存占用：15.4% (49.275GB/334.562GB)
[5][34] val done CPU 内存占用：16.0% (51.217GB/334.562GB)
[6][34] val done CPU 内存占用：16.4% (52.593GB/334.562GB)
[7][34] val done CPU 内存占用：16.8% (53.837GB/334.562GB)
[8][34] val done CPU 内存占用：17.0% (54.711GB/334.562GB)
[9][34] val done CPU 内存占用：17.3% (55.485GB/334.562GB)
[10][34] val done CPU 内存占用：17.6% (56.505GB/334.562GB)
[11][34] val done CPU 内存占用：17.8% (57.284GB/334.562GB)
[12][34] val done CPU 内存占用：18.2% (58.780GB/334.562GB)
[13][34] val done CPU 内存占用：18.3% (58.925GB/334.562GB)
[14][34] val done CPU 内存占用：18.5% (59.504GB/334.562GB)
[15][34] val done CPU 内存占用：18.6% (60.121GB/334.562GB)
[16][34] val done CPU 内存占用：18.9% (61.122GB/334.562GB)
[17][34] val done CPU 内存占用：19.2% (61.923GB/334.562GB)
[18][34] val done CPU 内存占用：19.4% (62.634GB/334.562GB)
[19][34] val done CPU 内存占用：19.7% (63.514GB/334.562GB)
[20][34] val done CPU 内存占用：20.0% (64.591GB/334.562GB)
[21][34] val done CPU 内存占用：20.2% (65.434GB/334.562GB)
[22][34] val done CPU 内存占用：20.4% (65.992GB/334.562GB)
[23][34] val done CPU 内存占用：20.5% (66.423GB/334.562GB)
[24][34] val done CPU 内存占用：20.7% (66.997GB/334.562GB)
[25][34] val done CPU 内存占用：20.8% (67.460GB/334.562GB)
[26][34] val done CPU 内存占用：21.0% (68.094GB/334.562GB)
[27][34] val done CPU 内存占用：21.0% (68.028GB/334.562GB)
[28][34] val done CPU 内存占用：21.1% (68.362GB/334.562GB)
[29][34] val done CPU 内存占用：21.2% (68.834GB/334.562GB)
[30][34] val done CPU 内存占用：21.4% (69.276GB/334.562GB)
[31][34] val done CPU 内存占用：21.4% (69.342GB/334.562GB)
[32][34] val done CPU 内存占用：21.5% (69.746GB/334.562GB)
[33][34] val done CPU 内存占用：21.6% (70.059GB/334.562GB)
[34][34] val done CPU 内存占用：21.7% (70.193GB/334.562GB)
[35][34] val done CPU 内存占用：21.8% (70.517GB/334.562GB)
[36][34] val done CPU 内存占用：21.8% (70.731GB/334.562GB)
[37][34] val done CPU 内存占用：21.9% (71.151GB/334.562GB)
[38][34] val done CPU 内存占用：22.0% (71.421GB/334.562GB)
[39][34] val done CPU 内存占用：22.0% (71.457GB/334.562GB)
[40][34] val done CPU 内存占用：22.1% (71.601GB/334.562GB)
[41][34] val done CPU 内存占用：22.1% (71.599GB/334.562GB)
[42][34] val done CPU 内存占用：22.2% (72.031GB/334.562GB)
[43][34] val done CPU 内存占用：22.1% (71.679GB/334.562GB)
[44][34] val done CPU 内存占用：22.1% (71.833GB/334.562GB)
[45][34] val done CPU 内存占用：22.2% (72.056GB/334.562GB)
[46][34] val done CPU 内存占用：22.4% (72.785GB/334.562GB)
[47][34] val done CPU 内存占用：22.5% (72.907GB/334.562GB)
[48][34] val done CPU 内存占用：22.4% (72.755GB/334.562GB)
[49][34] val done CPU 内存占用：22.5% (72.964GB/334.562GB)
all done CPU 内存占用：22.1% (71.763GB/334.562GB)

7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=C.UTF-8,Utf16=on,HugeFiles=on,64 bits,96 CPUs Intel(R) Xeon(R) CPU @ 2.00GHz (50653),ASM,AES-NI)

Scanning the drive:
1 folder, 2 files, 352359 bytes (345 KiB)

Creating archive: ./binctabl_10_target_5_period_point_v0_TPU.7z

Items to compress: 3


Files read from disk: 2
Archive size: 32646 bytes (32 KiB)
Everything is Ok
-------------------batch size: 1024 (0)-----------------------
-------------------batch size: 512 (1)-----------------------
E0710 04:51:33.625292439   61609 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T04:51:33.625275079+00:00", grpc_status:2}
E0710 04:51:33.625360296   61611 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:51:33.625344067+00:00"}
E0710 04:51:33.625418792   61613 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:51:33.625400573+00:00"}
E0710 04:51:33.625726266   62037 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T04:51:33.625710316+00:00"}
ddp: True
if_tqdm: False
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
learning_rate: 0.01664 -> 0.13312
batch_size: 1024
dataset length: 272955
dataloader length: 66
init train data done CPU 内存占用：3.6% (9.918GB/334.562GB)
broadcast_master_param
prepare done
each epoch step: 66
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank3]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank7]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank0]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank5]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[0][69] val done CPU 内存占用：9.0% (27.741GB/334.562GB)
[1][69] val done CPU 内存占用：9.9% (30.833GB/334.562GB)
[2][69] val done CPU 内存占用：10.2% (31.799GB/334.562GB)
[3][69] val done CPU 内存占用：10.6% (33.066GB/334.562GB)
[4][69] val done CPU 内存占用：10.9% (34.163GB/334.562GB)
[5][69] val done CPU 内存占用：11.3% (35.389GB/334.562GB)
[6][69] val done CPU 内存占用：11.5% (36.377GB/334.562GB)
[7][69] val done CPU 内存占用：11.9% (37.523GB/334.562GB)
[8][69] val done CPU 内存占用：12.0% (38.045GB/334.562GB)
[9][69] val done CPU 内存占用：12.3% (38.913GB/334.562GB)
[10][69] val done CPU 内存占用：12.5% (39.654GB/334.562GB)
[11][69] val done CPU 内存占用：12.8% (40.465GB/334.562GB)
[12][69] val done CPU 内存占用：13.0% (41.118GB/334.562GB)
[13][69] val done CPU 内存占用：13.1% (41.529GB/334.562GB)
[14][69] val done CPU 内存占用：13.3% (42.078GB/334.562GB)
[15][69] val done CPU 内存占用：13.4% (42.590GB/334.562GB)
[16][69] val done CPU 内存占用：13.6% (43.376GB/334.562GB)
[17][69] val done CPU 内存占用：13.8% (44.083GB/334.562GB)
[18][69] val done CPU 内存占用：14.1% (44.832GB/334.562GB)
[19][69] val done CPU 内存占用：14.2% (45.386GB/334.562GB)
[20][69] val done CPU 内存占用：14.3% (45.744GB/334.562GB)
[21][69] val done CPU 内存占用：14.5% (46.217GB/334.562GB)
[22][69] val done CPU 内存占用：14.7% (46.828GB/334.562GB)
[23][69] val done CPU 内存占用：14.8% (47.189GB/334.562GB)
[24][69] val done CPU 内存占用：14.9% (47.544GB/334.562GB)
[25][69] val done CPU 内存占用：14.9% (47.734GB/334.562GB)
[26][69] val done CPU 内存占用：15.1% (48.204GB/334.562GB)
[27][69] val done CPU 内存占用：15.2% (48.669GB/334.562GB)
[28][69] val done CPU 内存占用：15.3% (48.913GB/334.562GB)
[29][69] val done CPU 内存占用：15.4% (49.191GB/334.562GB)
[30][69] val done CPU 内存占用：15.4% (49.410GB/334.562GB)
[31][69] val done CPU 内存占用：15.5% (49.616GB/334.562GB)
[32][69] val done CPU 内存占用：15.6% (49.981GB/334.562GB)
[33][69] val done CPU 内存占用：15.6% (50.071GB/334.562GB)
[34][69] val done CPU 内存占用：15.7% (50.320GB/334.562GB)
[35][69] val done CPU 内存占用：15.7% (50.381GB/334.562GB)
[36][69] val done CPU 内存占用：15.8% (50.610GB/334.562GB)
[37][69] val done CPU 内存占用：15.9% (50.837GB/334.562GB)
[38][69] val done CPU 内存占用：15.9% (51.042GB/334.562GB)
[39][69] val done CPU 内存占用：15.9% (51.051GB/334.562GB)
[40][69] val done CPU 内存占用：16.0% (51.384GB/334.562GB)
[41][69] val done CPU 内存占用：16.0% (51.376GB/334.562GB)
[42][69] val done CPU 内存占用：16.1% (51.527GB/334.562GB)
[43][69] val done CPU 内存占用：16.1% (51.745GB/334.562GB)
[44][69] val done CPU 内存占用：16.2% (51.818GB/334.562GB)
[45][69] val done CPU 内存占用：16.2% (51.933GB/334.562GB)
[46][69] val done CPU 内存占用：16.2% (52.056GB/334.562GB)
[47][69] val done CPU 内存占用：16.3% (52.126GB/334.562GB)
[48][69] val done CPU 内存占用：16.3% (52.274GB/334.562GB)
[49][69] val done CPU 内存占用：16.3% (52.244GB/334.562GB)
all done CPU 内存占用：15.9% (51.032GB/334.562GB)

7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=C.UTF-8,Utf16=on,HugeFiles=on,64 bits,96 CPUs Intel(R) Xeon(R) CPU @ 2.00GHz (50653),ASM,AES-NI)

Scanning the drive:
1 folder, 2 files, 353623 bytes (346 KiB)

Creating archive: ./binctabl_10_target_5_period_point_v1_TPU.7z

Items to compress: 3


Files read from disk: 2
Archive size: 33655 bytes (33 KiB)
Everything is Ok
-------------------batch size: 512 (1)-----------------------
-------------------batch size: 256 (2)-----------------------
E0710 05:05:51.554528958  122802 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:05:51.554511227+00:00", grpc_status:2}
E0710 05:05:51.554605277  122800 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T05:05:51.554586586+00:00"}
E0710 05:05:51.554633871  122806 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:05:51.554617386+00:00", grpc_status:2}
E0710 05:05:51.554672879  122804 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:05:51.554657142+00:00", grpc_status:2}
ddp: True
if_tqdm: False
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
learning_rate: 0.01664 -> 0.13312
batch_size: 1024
dataset length: 272955
dataloader length: 133
init train data done CPU 内存占用：3.7% (10.058GB/334.562GB)
broadcast_master_param
prepare done
each epoch step: 133
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank4]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank0]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank2]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank7]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[0][139] val done CPU 内存占用：6.8% (20.449GB/334.562GB)
[1][139] val done CPU 内存占用：7.4% (22.647GB/334.562GB)
[2][139] val done CPU 内存占用：7.8% (23.698GB/334.562GB)
[3][139] val done CPU 内存占用：8.0% (24.655GB/334.562GB)
[4][139] val done CPU 内存占用：8.3% (25.514GB/334.562GB)
[5][139] val done CPU 内存占用：8.5% (26.323GB/334.562GB)
[6][139] val done CPU 内存占用：8.8% (27.111GB/334.562GB)
[7][139] val done CPU 内存占用：9.0% (27.959GB/334.562GB)
[8][139] val done CPU 内存占用：9.3% (28.907GB/334.562GB)
[9][139] val done CPU 内存占用：9.5% (29.484GB/334.562GB)
[10][139] val done CPU 内存占用：9.7% (30.189GB/334.562GB)
[11][139] val done CPU 内存占用：9.9% (30.860GB/334.562GB)
[12][139] val done CPU 内存占用：10.1% (31.396GB/334.562GB)
[13][139] val done CPU 内存占用：10.2% (31.772GB/334.562GB)
[14][139] val done CPU 内存占用：10.3% (32.235GB/334.562GB)
[15][139] val done CPU 内存占用：10.5% (32.722GB/334.562GB)
[16][139] val done CPU 内存占用：10.6% (33.109GB/334.562GB)
[17][139] val done CPU 内存占用：10.7% (33.494GB/334.562GB)
[18][139] val done CPU 内存占用：10.7% (33.676GB/334.562GB)
[19][139] val done CPU 内存占用：10.8% (33.874GB/334.562GB)
[20][139] val done CPU 内存占用：10.9% (34.221GB/334.562GB)
[21][139] val done CPU 内存占用：11.0% (34.566GB/334.562GB)
[22][139] val done CPU 内存占用：11.1% (34.879GB/334.562GB)
[23][139] val done CPU 内存占用：11.2% (35.260GB/334.562GB)
[24][139] val done CPU 内存占用：11.3% (35.718GB/334.562GB)
[25][139] val done CPU 内存占用：11.4% (35.916GB/334.562GB)
[26][139] val done CPU 内存占用：11.5% (36.249GB/334.562GB)
[27][139] val done CPU 内存占用：11.5% (36.177GB/334.562GB)
[28][139] val done CPU 内存占用：11.6% (36.427GB/334.562GB)
[29][139] val done CPU 内存占用：11.6% (36.460GB/334.562GB)
[30][139] val done CPU 内存占用：11.6% (36.619GB/334.562GB)
[31][139] val done CPU 内存占用：11.6% (36.713GB/334.562GB)
[32][139] val done CPU 内存占用：11.7% (36.746GB/334.562GB)
[33][139] val done CPU 内存占用：11.7% (36.750GB/334.562GB)
[34][139] val done CPU 内存占用：11.7% (36.802GB/334.562GB)
[35][139] val done CPU 内存占用：11.7% (36.956GB/334.562GB)
[36][139] val done CPU 内存占用：11.8% (37.062GB/334.562GB)
[37][139] val done CPU 内存占用：11.8% (37.183GB/334.562GB)
[38][139] val done CPU 内存占用：11.8% (37.377GB/334.562GB)
[39][139] val done CPU 内存占用：11.9% (37.546GB/334.562GB)
[40][139] val done CPU 内存占用：11.9% (37.518GB/334.562GB)
[41][139] val done CPU 内存占用：12.0% (37.733GB/334.562GB)
[42][139] val done CPU 内存占用：12.0% (37.879GB/334.562GB)
[43][139] val done CPU 内存占用：12.0% (37.919GB/334.562GB)
[44][139] val done CPU 内存占用：12.0% (38.028GB/334.562GB)
[45][139] val done CPU 内存占用：12.0% (37.990GB/334.562GB)
[46][139] val done CPU 内存占用：12.2% (38.730GB/334.562GB)
[47][139] val done CPU 内存占用：12.3% (38.854GB/334.562GB)
[48][139] val done CPU 内存占用：12.3% (39.032GB/334.562GB)
[49][139] val done CPU 内存占用：12.3% (39.027GB/334.562GB)
all done CPU 内存占用：11.9% (37.640GB/334.562GB)

7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=C.UTF-8,Utf16=on,HugeFiles=on,64 bits,96 CPUs Intel(R) Xeon(R) CPU @ 2.00GHz (50653),ASM,AES-NI)

Scanning the drive:
1 folder, 2 files, 355094 bytes (347 KiB)

Creating archive: ./binctabl_10_target_5_period_point_v2_TPU.7z

Items to compress: 3


Files read from disk: 2
Archive size: 34848 bytes (35 KiB)
Everything is Ok
-------------------batch size: 256 (2)-----------------------
-------------------batch size: 128 (3)-----------------------
E0710 05:26:31.554721851  184295 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:26:31.554698277+00:00", grpc_status:2}
E0710 05:26:31.554862591  184297 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {grpc_status:2, created_time:"2024-07-10T05:26:31.554846457+00:00"}
E0710 05:26:31.554975458  184719 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:26:31.554956461+00:00", grpc_status:2}
E0710 05:26:31.554984369  184293 oauth2_credentials.cc:238]            oauth_fetch: UNKNOWN:C-ares status is not ARES_SUCCESS qtype=A name=metadata.google.internal. is_balancer=0: Domain name not found {created_time:"2024-07-10T05:26:31.554969781+00:00", grpc_status:2}
ddp: True
if_tqdm: False
[W socket.cpp:697] [c10d] The client socket has failed to connect to [localhost]:12355 (errno: 99 - Cannot assign requested address).
learning_rate: 0.01664 -> 0.13312
batch_size: 1024
dataset length: 272955
dataloader length: 266
init train data done CPU 内存占用：3.7% (10.238GB/334.562GB)
broadcast_master_param
prepare done
each epoch step: 266
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.10/site-packages/torch/autograd/__init__.py:266: UserWarning: aten::reshape: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank3]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank6]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank0]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[rank4]:[W logger.cpp:337] Warning: Time stats are currently only collected for CPU and CUDA devices. Please refer to CpuTimer or CudaTimer for how to register timer for other device type. (function operator())
[0][279] val done CPU 内存占用：5.7% (16.745GB/334.562GB)
[1][279] val done CPU 内存占用：6.1% (18.054GB/334.562GB)
[2][279] val done CPU 内存占用：6.3% (18.661GB/334.562GB)
[3][279] val done CPU 内存占用：6.4% (19.123GB/334.562GB)
[4][279] val done CPU 内存占用：6.5% (19.561GB/334.562GB)
[5][279] val done CPU 内存占用：6.6% (19.956GB/334.562GB)
[6][279] val done CPU 内存占用：6.7% (20.324GB/334.562GB)
[7][279] val done CPU 内存占用：6.9% (20.767GB/334.562GB)
[8][279] val done CPU 内存占用：6.9% (20.991GB/334.562GB)
[9][279] val done CPU 内存占用：7.1% (21.340GB/334.562GB)
[10][279] val done CPU 内存占用：7.1% (21.582GB/334.562GB)
[11][279] val done CPU 内存占用：7.2% (21.872GB/334.562GB)
[12][279] val done CPU 内存占用：7.3% (22.140GB/334.562GB)
[13][279] val done CPU 内存占用：7.4% (22.365GB/334.562GB)
[14][279] val done CPU 内存占用：7.4% (22.485GB/334.562GB)
[15][279] val done CPU 内存占用：7.4% (22.591GB/334.562GB)
[16][279] val done CPU 内存占用：7.5% (22.692GB/334.562GB)
[17][279] val done CPU 内存占用：7.5% (22.776GB/334.562GB)
[18][279] val done CPU 内存占用：7.5% (22.854GB/334.562GB)
[19][279] val done CPU 内存占用：7.5% (22.929GB/334.562GB)
[20][279] val done CPU 内存占用：7.6% (23.085GB/334.562GB)
[21][279] val done CPU 内存占用：7.6% (23.203GB/334.562GB)
[22][279] val done CPU 内存占用：7.6% (23.295GB/334.562GB)
[23][279] val done CPU 内存占用：7.8% (23.862GB/334.562GB)
[24][279] val done CPU 内存占用：7.9% (24.065GB/334.562GB)
[25][279] val done CPU 内存占用：7.9% (24.260GB/334.562GB)
[26][279] val done CPU 内存占用：8.0% (24.418GB/334.562GB)
[27][279] val done CPU 内存占用：8.0% (24.589GB/334.562GB)
[28][279] val done CPU 内存占用：8.1% (24.732GB/334.562GB)
[29][279] val done CPU 内存占用：8.1% (24.878GB/334.562GB)
[30][279] val done CPU 内存占用：8.2% (25.111GB/334.562GB)
[31][279] val done CPU 内存占用：8.2% (25.236GB/334.562GB)
[32][279] val done CPU 内存占用：8.3% (25.362GB/334.562GB)
[33][279] val done CPU 内存占用：8.3% (25.465GB/334.562GB)
[34][279] val done CPU 内存占用：8.3% (25.612GB/334.562GB)
[35][279] val done CPU 内存占用：8.4% (25.717GB/334.562GB)
[36][279] val done CPU 内存占用：8.4% (25.899GB/334.562GB)
[37][279] val done CPU 内存占用：8.4% (25.991GB/334.562GB)
[38][279] val done CPU 内存占用：8.5% (26.045GB/334.562GB)
[39][279] val done CPU 内存占用：8.5% (26.053GB/334.562GB)
[40][279] val done CPU 内存占用：8.5% (26.084GB/334.562GB)
[41][279] val done CPU 内存占用：8.5% (26.155GB/334.562GB)
[42][279] val done CPU 内存占用：8.5% (26.163GB/334.562GB)
[43][279] val done CPU 内存占用：8.5% (26.189GB/334.562GB)
[44][279] val done CPU 内存占用：8.5% (26.251GB/334.562GB)
[45][279] val done CPU 内存占用：8.5% (26.326GB/334.562GB)
[46][279] val done CPU 内存占用：8.5% (26.353GB/334.562GB)
[47][279] val done CPU 内存占用：8.6% (26.384GB/334.562GB)
[48][279] val done CPU 内存占用：8.6% (26.417GB/334.562GB)
[49][279] val done CPU 内存占用：8.5% (26.158GB/334.562GB)
all done CPU 内存占用：8.0% (24.474GB/334.562GB)

7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=C.UTF-8,Utf16=on,HugeFiles=on,64 bits,96 CPUs Intel(R) Xeon(R) CPU @ 2.00GHz (50653),ASM,AES-NI)

Scanning the drive:
1 folder, 2 files, 356402 bytes (349 KiB)

Creating archive: ./binctabl_10_target_5_period_point_v3_TPU.7z

Items to compress: 3


Files read from disk: 2
Archive size: 35591 bytes (35 KiB)
Everything is Ok
-------------------batch size: 128 (3)-----------------------
-------------------batch size: 64 (4)-----------------------

"""

def yfunc_target_long_short(x):
    # long/ short
    x1, x2 = x
    if x1 > 0:# 多头盈利
        return 0
    elif x2 > 0:# 空头盈利
        return 1
    else:
        return 2

def yfunc_target_simple(x):
    if x > 0:
        return 0
    elif x < 0:
        return 1
    else:
        return 2

class test(test_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        symbols = ['ETHFDUSD', 'ETHUSDT', 'BTCFDUSD', 'BTCUSDT']

        vars = []
        for classify_idx, name, yfunc in zip(
            [[0, 1], 2, 3], 
            ['10_target_long_short', '10_target_mid', '10_target_5_period_point'],
            [yfunc_target_long_short, yfunc_target_simple, yfunc_target_simple]
        ):
            vars.append((classify_idx, name, yfunc))

        classify_idx, targrt_name, yfunc = vars[2]
        self.y_n = 3

        batch_n = 16 * 8
        title = f'binctabl_{targrt_name}_v{self.idx}'
        data_parm = {
            'predict_n': [10, 20, 30],
            'pass_n': 100,
            'y_n': self.y_n,
            'begin_date': '2024-05-01',
            'data_rate': (2, 2, 2),
            'total_hours': int(2*3),
            'symbols': '@'.join(symbols),
            'target': targrt_name,
            'std_mode': '5d'  # 4h/1d/5d
        }

        # 实例化 参数对象
        self.para = Params(
            train_title=title, root=f'./{title}', data_set=f'{data_parm2str(data_parm)}.7z',
            learning_rate=0.00013*batch_n, batch_size=64*batch_n, epochs=10,

            # 数据增强
            random_scale=0.05, random_mask_row=0.7,

            # 每4个样本取一个数据
            down_freq=8,

            # 3分类
            classify=True,
            y_n=self.y_n, classify_y_idx=classify_idx, y_func=yfunc,

            data_folder=self.data_folder,

            describe=f'target={targrt_name}',
            amp=self.amp
        )

    # 初始化模型
    # 返回一个 torch model
    def get_model(self):
        return m_bin_ctabl(60, 40, 100, 40, 120, 10, self.y_n, 1)

    def get_transform(self, device):
        return transform(device, self.para, 105)