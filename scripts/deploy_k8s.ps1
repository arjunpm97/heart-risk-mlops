# Deploy the API + monitoring to a local cluster (Docker Desktop Kubernetes or Minikube).
# Usage:  .\scripts\deploy_k8s.ps1            # Docker Desktop
#         .\scripts\deploy_k8s.ps1 -Minikube  # Minikube
param([switch]$Minikube, [string]$Image = "heart-risk-api:1.0.0")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if ($Minikube) {
    Write-Host ">> Building image inside minikube's Docker daemon"
    & minikube -p minikube docker-env --shell powershell | Invoke-Expression
    minikube addons enable ingress | Out-Null
    minikube addons enable metrics-server | Out-Null
}
Write-Host ">> docker build -t $Image ."
docker build -t $Image .

Write-Host ">> Applying manifests"
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
kubectl -n heart-risk create configmap grafana-dashboards `
    --from-file=monitoring/grafana/dashboards/heart_risk_dashboard.json `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/

Write-Host ">> Waiting for rollout"
kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s
kubectl -n heart-risk get pods,svc,ingress,hpa -o wide

if ($Minikube) {
    Write-Host "Run 'minikube tunnel' in another terminal to get the LoadBalancer IP, or:"
    Write-Host "  minikube service heart-risk-api -n heart-risk --url"
} else {
    Write-Host "API (LoadBalancer): http://localhost/docs"
}
Write-Host "Grafana: kubectl -n heart-risk port-forward svc/grafana 3000:3000  -> http://localhost:3000"
Write-Host "Prometheus: kubectl -n heart-risk port-forward svc/prometheus 9090:9090"
