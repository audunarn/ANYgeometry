$ErrorActionPreference='Stop'
$repo='C:/Github/ANYgeometry/.worktrees/intersection-continuation'
$runtime='C:/Users/AudunArnesenNyhus/AppData/Local/Temp/anygeometry-arc-37d7342-20261006/env/Scripts/python.exe'
$prior=12.6039208
$attempt=Join-Path $repo ('reports/coons-extrusion-support-proof-01/root-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
New-Item -ItemType Directory -Path $attempt | Out-Null
$env:PYTHONPATH=Join-Path $repo 'src'
$env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$env:PROOF_RUNTIME=Join-Path $attempt 'runtime.json';$env:PROOF_XML=Join-Path $attempt 'results.xml'
$hashes=@{}
foreach($f in @('src/anygeometry/native_support_snapshots.py','src/anygeometry/native_material_reference_scope.py','src/anygeometry/polynomial_extrusion_support.py','tests/test_polynomial_extrusion_support.py')){$hashes[$f]=(Get-FileHash (Join-Path $repo $f)).Hash.ToLower()}
$watch=[Diagnostics.Stopwatch]::StartNew()
$p=Start-Process -FilePath $runtime -ArgumentList @('-X','utf8','reports/coons-extrusion-support-proof-01/public-check.py') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $attempt 'stdout.log') -RedirectStandardError (Join-Path $attempt 'stderr.log')
$done=$p.WaitForExit([Math]::Max(1,2250-[int]$watch.Elapsed.TotalMilliseconds))
if(-not $done){$p.Kill($true);$p.WaitForExit()}
$watch.Stop()
$m=[ordered]@{head='740944fbd9801f264e84a1bcd32ed4b57103e1bb';source_hashes=$hashes;command=@($runtime,'-X','utf8','reports/coons-extrusion-support-proof-01/public-check.py');timeout_seconds=2.25;process_wall_seconds=$watch.Elapsed.TotalSeconds;cumulative_seconds=($prior+$watch.Elapsed.TotalSeconds);remaining_seconds=(15-$prior-$watch.Elapsed.TotalSeconds);exit_code=$p.ExitCode;timed_out=(-not $done);status=if($done -and $p.ExitCode -eq 0){'passed'}else{'failed-or-incomplete'};memory_measurement='unavailable; not measured'}
if(Test-Path $env:PROOF_XML){[xml]$x=Get-Content $env:PROOF_XML -Raw;$m.test_counts=@{tests=$x.testsuites.testsuite.tests;failures=$x.testsuites.testsuite.failures;errors=$x.testsuites.testsuite.errors;skipped=$x.testsuites.testsuite.skipped}}
$m | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $attempt 'metadata.json')
Write-Output ($m | ConvertTo-Json -Depth 6 -Compress)
Write-Output $attempt

