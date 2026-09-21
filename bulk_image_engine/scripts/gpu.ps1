<#
.SYNOPSIS
  Open and close a GPU instance for a render job.

.DESCRIPTION
  launch     Start a g5.xlarge that installs, renders, uploads to S3 and then
             TERMINATES ITSELF. A finished or failed run cannot leave a GPU
             billing overnight.
  status     Show every running instance in the region (not just ours).
  logs       Tail the render log from the instance console output.
  stop       Stop (keeps the disk and the 7 GB model cache).
  start      Start a stopped instance again.
  terminate  Delete it. Nothing billing afterwards.

.NOTES
  NOT YET TESTED - the account's GPU quota is still 0 and no instance has ever
  been launched. Treat the first run as a dry run and watch it closely.

  Requires: aws sso login --profile ashish-admin

.EXAMPLE
  .\gpu.ps1 launch -Batches 15 -Bucket my-image-output -Prompts prompts.csv
  .\gpu.ps1 status
  .\gpu.ps1 terminate
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory, Position = 0)]
  [ValidateSet("launch", "status", "logs", "stop", "start", "terminate")]
  [string]$Action,

  [string]$Profile      = "ashish-admin",
  [string]$Region       = "us-east-1",
  [string]$InstanceType = "g5.xlarge",
  [string]$KeyName      = "bulk-image-key",
  [string]$SgName       = "bulk-image-sg",
  [string]$Tag          = "bulk-image-engine",
  [string]$Repo         = "https://github.com/Ashish4283/Bulk_AI_Image_Generation.git",
  [string]$Prompts      = "sample_prompts.csv",
  [int]   $Batches      = 3,
  [int]   $Size         = 1024,
  [string]$Bucket       = "",          # required for launch: where zips are uploaded
  [switch]$Spot,                       # ~1/3 the price; safe because renders resume
  [string]$InstanceId   = ""
)

# "Continue", not "Stop": PowerShell turns any stderr output from a native exe
# into a NativeCommandError, so "Stop" would abort on AWS CLI warnings and
# bury our own error messages. We check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"
$AWS = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $AWS)) { $AWS = "aws" }
$Common = @("--profile", $Profile, "--region", $Region)

function Invoke-Aws { & $AWS @args @Common 2>&1 }

function Invoke-AwsChecked {
  $out = & $AWS @args @Common 2>&1 | Out-String
  if ($LASTEXITCODE -ne 0) {
    if ($out -match "expired|SSO session") {
      throw "AWS session expired. Run:  aws sso login --profile $Profile"
    }
    throw "aws $($args -join ' ') failed:`n$($out.Trim())"
  }
  return $out.Trim()
}

function Get-OurInstance {
  if ($InstanceId) { return $InstanceId }
  $id = Invoke-AwsChecked ec2 describe-instances `
    --filters "Name=tag:Project,Values=$Tag" "Name=instance-state-name,Values=pending,running,stopping,stopped" `
    --query "Reservations[].Instances[0].InstanceId" --output text
  if (-not $id -or $id -eq "None") { return $null }
  return ($id -split "\s+")[0]
}

function Assert-Login {
  $who = & $AWS sts get-caller-identity @Common --query Arn --output text 2>&1 | Out-String
  if ($LASTEXITCODE -ne 0) {
    if ($who -match "expired|SSO session") {
      throw "AWS session expired. Run:  aws sso login --profile $Profile"
    }
    throw "Not signed in to AWS ($($who.Trim())). Run:  aws sso login --profile $Profile"
  }
  Write-Host "signed in as $($who.Trim())" -ForegroundColor DarkGray
}

switch ($Action) {

  "launch" {
    Assert-Login
    if (-not $Bucket) { throw "-Bucket is required: results are uploaded there before the instance self-terminates" }

    # Refuse to start a second instance by accident - that is double billing.
    $existing = Get-OurInstance
    if ($existing) { throw "instance $existing already exists. Use 'status', or 'terminate' first." }

    $quota = Invoke-Aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query "Quota.Value" --output text
    Write-Host "GPU vCPU quota: $quota" -ForegroundColor DarkGray
    if ([double]$quota -lt 4) { throw "GPU quota is $quota vCPUs. g5.xlarge needs 4. Wait for the AWS case to be approved." }

    # Deep Learning AMI: driver and CUDA preinstalled.
    $ami = Invoke-Aws ssm get-parameter `
      --name "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id" `
      --query "Parameter.Value" --output text
    Write-Host "AMI: $ami"

    # Key pair
    $haveKey = Invoke-Aws ec2 describe-key-pairs --key-names $KeyName --query "KeyPairs[0].KeyName" --output text 2>$null
    if (-not $haveKey -or $haveKey -eq "None") {
      $pem = Join-Path $PSScriptRoot "$KeyName.pem"
      Invoke-Aws ec2 create-key-pair --key-name $KeyName --query "KeyMaterial" --output text | Set-Content $pem -Encoding ascii
      Write-Host "created key pair -> $pem  (keep it; it is the only copy)" -ForegroundColor Yellow
    }

    # Security group, SSH open only to this machine's public IP
    $sg = Invoke-Aws ec2 describe-security-groups --group-names $SgName --query "SecurityGroups[0].GroupId" --output text 2>$null
    if (-not $sg -or $sg -eq "None") {
      $sg = Invoke-Aws ec2 create-security-group --group-name $SgName --description "bulk image engine" --query "GroupId" --output text
      $myIp = (Invoke-RestMethod "https://checkip.amazonaws.com").Trim()
      Invoke-Aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$myIp/32" | Out-Null
      Write-Host "created security group $sg (SSH from $myIp only)"
    }

    # user-data: install, render, upload, self-terminate.
    $userData = @"
#!/bin/bash
set -x
exec > /var/log/render.log 2>&1
cd /home/ubuntu
sudo -u ubuntu git clone $Repo app
cd app/bulk_image_engine
sudo -u ubuntu python3 -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
sudo -u ubuntu python3 -m pip install --quiet -r requirements.txt
sudo -u ubuntu python3 engine.py --prompts $Prompts --all --batches $Batches --size $Size --bundle
aws s3 cp output/ s3://$Bucket/`$(date +%Y%m%d-%H%M%S)/ --recursive --exclude "*" --include "*.zip"
# Close the instance no matter how the render ended.
shutdown -h now
"@
    $udFile = Join-Path $env:TEMP "gpu-userdata.sh"
    [IO.File]::WriteAllText($udFile, $userData.Replace("`r`n", "`n"))

    $marketOpts = @()
    if ($Spot) { $marketOpts = @("--instance-market-options", "MarketType=spot") }

    Write-Host "launching $InstanceType ..." -ForegroundColor Cyan
    $id = Invoke-Aws ec2 run-instances `
      --image-id $ami --instance-type $InstanceType --key-name $KeyName `
      --security-group-ids $sg --count 1 `
      --instance-initiated-shutdown-behavior terminate `
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}" `
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$Tag}]" `
      --user-data "file://$udFile" `
      @marketOpts `
      --query "Instances[0].InstanceId" --output text

    Write-Host "launched $id" -ForegroundColor Green
    Write-Host "It will render $Batches batch(es), upload zips to s3://$Bucket/, then TERMINATE ITSELF."
    Write-Host "Watch:  .\gpu.ps1 status        Logs:  .\gpu.ps1 logs"
    Write-Host "If anything looks wrong:  .\gpu.ps1 terminate" -ForegroundColor Yellow
  }

  "status" {
    Assert-Login
    Write-Host "`n--- every running/stopped instance in $Region ---"
    Invoke-Aws ec2 describe-instances `
      --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" `
      --query "Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,IP:PublicIpAddress,Launched:LaunchTime}" `
      --output table
    Write-Host "(empty means nothing is billing)" -ForegroundColor DarkGray
  }

  "logs" {
    Assert-Login
    $id = Get-OurInstance; if (-not $id) { throw "no instance found" }
    Invoke-Aws ec2 get-console-output --instance-id $id --output text |
      Select-String -Pattern "Rendered|batch|Error|error|fatal" | Select-Object -Last 40
  }

  "stop" {
    Assert-Login
    $id = Get-OurInstance; if (-not $id) { throw "no instance found" }
    Invoke-Aws ec2 stop-instances --instance-ids $id --query "StoppingInstances[0].CurrentState.Name" --output text
    Write-Host "stopped $id (disk and model cache kept; EBS still bills a few pennies/day)"
  }

  "start" {
    Assert-Login
    $id = Get-OurInstance; if (-not $id) { throw "no instance found" }
    Invoke-Aws ec2 start-instances --instance-ids $id --query "StartingInstances[0].CurrentState.Name" --output text
  }

  "terminate" {
    Assert-Login
    $id = Get-OurInstance; if (-not $id) { Write-Host "no instance found - nothing to terminate"; break }
    Invoke-Aws ec2 terminate-instances --instance-ids $id --query "TerminatingInstances[0].CurrentState.Name" --output text
    Write-Host "terminating $id" -ForegroundColor Green
    Write-Host "confirm with:  .\gpu.ps1 status"
  }
}
