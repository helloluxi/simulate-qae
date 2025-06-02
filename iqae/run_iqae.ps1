
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_CMAKE -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
set-alias run .\build\Release\iqae

if (Test-Path iqae.txt) {
    Remove-Item iqae.txt
    Write-Host "Removed old iqae.txt" -ForegroundColor Yellow
}
16,32,64,128,512,1024 | % {
    run iqae $(1.0/$_) 0.05 32 >> iqae.txt
}

Write-Host "OK: iqae.txt" -ForegroundColor Green

# if (Test-Path chebae.txt) {
#     Remove-Item chebae.txt
#     Write-Host "Removed old chebae.txt" -ForegroundColor Yellow
# }
# 16,32,64,128,512,1024 | % {
#     run chebae $(1.0/$_) 0.01 32 >> chebae.txt
# }

# Write-Host "OK: chebae.txt" -ForegroundColor Green