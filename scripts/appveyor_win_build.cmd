cd build
SET PGO=false
IF %APPVEYOR_REPO_TAG%==true IF %DX%==false IF %ONNX%==false SET PGO=true
IF %PGO%==true meson configure -Db_pgo=generate
ninja
IF EXIST lc0.pdb del lc0.pdb
IF ERRORLEVEL 1 EXIT
IF %NAME%==cpu-openblas copy C:\cache\OpenBLAS\dist64\bin\libopenblas.dll
IF %NAME%==cpu-dnnl copy C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %NAME%==onednn copy C:\cache\%DNNL_NAME%\bin\dnnl.dll
IF %NAME%==onednn copy dnnl.dll ..
copy "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-override.dll
copy "%MIMALLOC_PATH%"\out\msvc-x64\Release\mimalloc-redirect.dll
IF %PGO%==true (
  IF %OPENCL%==true copy C:\cache\opencl-nug.0.777.77\build\native\bin\OpenCL.dll
  IF %CUDA%==true copy "%CUDA_PATH%"\bin\*.dll
  IF %CUDNN%==true copy "%CUDA_PATH%"\cuda\bin\cudnn64_7.dll
  lc0 benchmark --num-positions=1 --backend=trivial --movetime=10000
  meson configure -Db_pgo=use
  ninja 
)
IF %NAME%==onnx (
  ren lc0.exe lc0-trt.exe
  meson configure -Ddefault_backend= -Dcudnn_libdirs= -Dgtest=%GTEST%
  ninja
  ren lc0.exe lc0-dml.exe
)
cd ..
