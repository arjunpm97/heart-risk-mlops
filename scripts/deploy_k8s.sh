#!/usr/bin/env bash
# Deploy the API + monitoring to a local cluster. Usage: ./scripts/deploy_k8s.sh [minikube]
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-heart-risk-api:1.0.0}"

if [[ "${1:-}" == "minikube" ]]; then
  eval "$(minikube -p minikube docker-env)"
  minikube addons enable ingress >/dev/null
  minikube addons enable metrics-server >/dev/null
fi

docker build -t "$IMAGE" .
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
kubectl -n heart-risk create configmap grafana-dashboards \
  --from-file=monitoring/grafana/dashboards/heart_risk_dashboard.json \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/
kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s
kubectl -n heart-risk get pods,svc,ingress,hpa -o wide
