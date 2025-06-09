# 🚀 Kelpie Carbon v1: Deployment Guide

## **Overview**

This guide covers deployment strategies for the Kelpie Carbon v1 application across different environments: local development, staging, and production. The application is designed to be cloud-native and can be deployed using various containerization and orchestration platforms.

## **Deployment Environments**

### **Local Development**
- **Purpose**: Development and testing
- **Requirements**: Python 3.12+, Poetry, modern browser
- **Performance**: Single-user, development features enabled

### **Staging**
- **Purpose**: Pre-production testing and QA
- **Requirements**: Docker, container orchestration
- **Performance**: Multi-user, production-like configuration

### **Production**
- **Purpose**: Live application serving end users
- **Requirements**: High availability, monitoring, scaling
- **Performance**: Auto-scaling, load balancing, monitoring

---

## **Local Development Deployment**

### **Prerequisites**
```bash
# Install Python 3.12+
python --version  # Should be 3.12 or higher

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### **Setup Process**
```bash
# 1. Clone repository
git clone https://github.com/your-org/kelpie-carbon-v1.git
cd kelpie-carbon-v1

# 2. Install dependencies
poetry install

# 3. Activate virtual environment
poetry shell

# 4. Run development server
poetry run uvicorn src.kelpie_carbon_v1.api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Access application
# Open http://localhost:8000 in your browser
```

### **Development Configuration**
```bash
# Environment variables for development
export KELPIE_ENV=development
export KELPIE_DEBUG=true
export KELPIE_LOG_LEVEL=debug
export KELPIE_CACHE_SIZE=100
```

### **Development Tools**
```bash
# Run tests
poetry run pytest

# Code formatting
poetry run black src/ tests/

# Linting
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/

# Security scanning
poetry run bandit -r src/
```

---

## **Docker Deployment**

### **Dockerfile**
```dockerfile
# Multi-stage build for production optimization
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KELPIE_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash kelpie

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY --chown=kelpie:kelpie . .

# Switch to non-root user
USER kelpie

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.kelpie_carbon_v1.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Docker Compose for Development**
```yaml
# docker-compose.yml
version: '3.8'

services:
  kelpie-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - KELPIE_ENV=development
      - KELPIE_DEBUG=true
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    command: uvicorn src.kelpie_carbon_v1.api.main:app --host 0.0.0.0 --port 8000 --reload
    
  # Optional: Redis for caching (future enhancement)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### **Building and Running**
```bash
# Build Docker image
docker build -t kelpie-carbon-v1:latest .

# Run container
docker run -d \
  --name kelpie-app \
  -p 8000:8000 \
  -e KELPIE_ENV=production \
  kelpie-carbon-v1:latest

# View logs
docker logs -f kelpie-app

# Run with Docker Compose
docker-compose up -d

# Scale application (if using multiple replicas)
docker-compose up -d --scale kelpie-app=3
```

---

## **Cloud Deployment**

### **AWS Deployment**

#### **ECS (Elastic Container Service)**
```yaml
# aws-ecs-task-definition.json
{
  "family": "kelpie-carbon-v1",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "kelpie-app",
      "image": "your-ecr-repo/kelpie-carbon-v1:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "KELPIE_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/kelpie-carbon-v1",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### **CloudFormation Template**
```yaml
# aws-infrastructure.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Kelpie Carbon v1 Infrastructure'

Parameters:
  ImageURI:
    Type: String
    Description: ECR Image URI
  
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID for deployment
  
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs for deployment

Resources:
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: kelpie-carbon-v1
      CapacityProviders:
        - FARGATE
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: kelpie-carbon-v1-alb
      Scheme: internet-facing
      Type: application
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: kelpie-carbon-v1-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref AppSecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: kelpie-app
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup

  # Auto Scaling
  ServiceScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 10
      MinCapacity: 2
      ResourceId: !Sub "service/${ECSCluster}/${ECSService.Name}"
      RoleARN: !Sub "arn:aws:iam::${AWS::AccountId}:role/aws-service-role/ecs.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_ECSService"
      ScalableDimension: ecs:service:DesiredCount
      ServiceNamespace: ecs

  ServiceScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: kelpie-carbon-v1-scaling-policy
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref ServiceScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ECSServiceAverageCPUUtilization
        TargetValue: 70.0
        ScaleOutCooldown: 300
        ScaleInCooldown: 300

Outputs:
  LoadBalancerDNS:
    Description: Load Balancer DNS Name
    Value: !GetAtt LoadBalancer.DNSName
    Export:
      Name: !Sub "${AWS::StackName}-LoadBalancerDNS"
```

### **Google Cloud Platform (GCP)**

#### **Cloud Run Deployment**
```yaml
# gcp-cloudrun.yml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: kelpie-carbon-v1
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/ingress-status: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "1Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/kelpie-carbon-v1:latest
        ports:
        - containerPort: 8000
        env:
        - name: KELPIE_ENV
          value: "production"
        resources:
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          timeoutSeconds: 10
```

#### **Deployment Script**
```bash
#!/bin/bash
# deploy-gcp.sh

PROJECT_ID="your-project-id"
SERVICE_NAME="kelpie-carbon-v1"
REGION="us-central1"

# Build and push Docker image
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars KELPIE_ENV=production

# Get service URL
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
```

### **Azure Deployment**

#### **Container Instances**
```yaml
# azure-container-instance.yml
apiVersion: 2019-12-01
location: East US
name: kelpie-carbon-v1
properties:
  containers:
  - name: kelpie-app
    properties:
      image: youracr.azurecr.io/kelpie-carbon-v1:latest
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 1.5
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: KELPIE_ENV
        value: production
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 30
        timeoutSeconds: 10
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
    dnsNameLabel: kelpie-carbon-v1
```

---

## **Kubernetes Deployment**

### **Kubernetes Manifests**
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kelpie-carbon-v1
  labels:
    app: kelpie-carbon-v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kelpie-carbon-v1
  template:
    metadata:
      labels:
        app: kelpie-carbon-v1
    spec:
      containers:
      - name: kelpie-app
        image: kelpie-carbon-v1:latest
        ports:
        - containerPort: 8000
        env:
        - name: KELPIE_ENV
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: kelpie-carbon-v1-service
spec:
  selector:
    app: kelpie-carbon-v1
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: kelpie-carbon-v1-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - kelpie.your-domain.com
    secretName: kelpie-tls
  rules:
  - host: kelpie.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kelpie-carbon-v1-service
            port:
              number: 80
```

### **Helm Chart**
```yaml
# helm/values.yml
replicaCount: 3

image:
  repository: kelpie-carbon-v1
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: kelpie.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: kelpie-tls
      hosts:
        - kelpie.your-domain.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

nodeSelector: {}
tolerations: []
affinity: {}
```

---

## **Configuration Management**

### **Environment Variables**
```bash
# Production environment variables
KELPIE_ENV=production
KELPIE_DEBUG=false
KELPIE_LOG_LEVEL=info
KELPIE_CACHE_SIZE=1000
KELPIE_MAX_WORKERS=4

# Optional: Database configuration (future enhancement)
DATABASE_URL=postgresql://user:pass@localhost/kelpie
REDIS_URL=redis://localhost:6379

# Optional: External service configuration
PLANETARY_COMPUTER_SUBSCRIPTION_KEY=your-key
SENTRY_DSN=your-sentry-dsn

# Security configuration
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=your-domain.com,localhost
```

### **Configuration Files**
```yaml
# config/production.yml
app:
  name: "Kelpie Carbon v1"
  version: "1.0.0"
  environment: "production"
  debug: false

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

cache:
  size: 1000
  ttl: 3600

logging:
  level: "info"
  format: "json"
  
monitoring:
  enabled: true
  metrics_port: 9090
```

---

## **Monitoring and Observability**

### **Health Checks**
```python
# Add to src/kelpie_carbon_v1/api/main.py
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("KELPIE_ENV", "development")
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if application is ready to serve traffic
    # e.g., check database connections, external services
    return {"status": "ready"}
```

### **Prometheus Metrics**
```python
# Optional: Add Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('kelpie_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('kelpie_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Collect Prometheus metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### **Logging Configuration**
```python
# src/kelpie_carbon_v1/core/logging.py
import logging
import sys
from typing import Dict, Any

def setup_logging(level: str = "info", format_type: str = "json") -> None:
    """Setup structured logging."""
    
    if format_type == "json":
        import json_logging
        json_logging.init_fastapi(enable_json=True)
        
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "json": {
                "()": "json_logging.JSONLogFormatter",
            },
        },
        "handlers": {
            "default": {
                "formatter": format_type,
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
        },
        "root": {
            "level": level.upper(),
            "handlers": ["default"],
        },
    }
    
    logging.config.dictConfig(logging_config)
```

---

## **Security Configuration**

### **TLS/SSL Configuration**
```nginx
# nginx configuration for HTTPS termination
server {
    listen 443 ssl http2;
    server_name kelpie.your-domain.com;
    
    ssl_certificate /etc/ssl/certs/kelpie.crt;
    ssl_certificate_key /etc/ssl/private/kelpie.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://kelpie-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

upstream kelpie-backend {
    server 127.0.0.1:8000;
}
```

### **Security Headers**
```python
# Add security middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["kelpie.your-domain.com", "localhost"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kelpie.your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## **Backup and Disaster Recovery**

### **Data Backup Strategy**
```bash
#!/bin/bash
# backup-script.sh

# Backup analysis results (if using persistent storage)
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup application logs
cp -r /var/log/kelpie $BACKUP_DIR/logs

# Backup configuration
cp -r /etc/kelpie $BACKUP_DIR/config

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://kelpie-backups/$(date +%Y%m%d)/

# Cleanup old backups (keep 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} +
```

### **Disaster Recovery Plan**
1. **RTO (Recovery Time Objective)**: 15 minutes
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Backup Frequency**: Daily automated backups
4. **Recovery Testing**: Monthly recovery drills

---

## **Performance Optimization**

### **Production Tuning**
```python
# Production server configuration
uvicorn src.kelpie_carbon_v1.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --preload \
  --access-log \
  --use-colors
```

### **Caching Configuration**
```python
# Redis caching (optional future enhancement)
import redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis_client = redis.from_url("redis://localhost:6379")
    FastAPICache.init(RedisBackend(redis_client), prefix="kelpie-cache")
```

---

## **Troubleshooting**

### **Common Deployment Issues**

#### **Container Startup Issues**
```bash
# Check container logs
docker logs kelpie-app

# Check resource usage
docker stats kelpie-app

# Exec into container for debugging
docker exec -it kelpie-app /bin/bash
```

#### **Network Connectivity Issues**
```bash
# Test internal connectivity
kubectl exec -it deployment/kelpie-carbon-v1 -- curl localhost:8000/health

# Check service endpoints
kubectl get endpoints kelpie-carbon-v1-service

# Test external connectivity
curl -I https://kelpie.your-domain.com/health
```

#### **Performance Issues**
```bash
# Check resource utilization
kubectl top pods -l app=kelpie-carbon-v1

# Scale up if needed
kubectl scale deployment kelpie-carbon-v1 --replicas=5

# Check logs for errors
kubectl logs -f deployment/kelpie-carbon-v1 --tail=100
```

### **Monitoring and Alerting**
```yaml
# alertmanager-rules.yml
groups:
- name: kelpie-alerts
  rules:
  - alert: KelpieHighMemoryUsage
    expr: container_memory_usage_bytes{pod=~"kelpie-.*"} / container_spec_memory_limit_bytes > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage in Kelpie application"
      
  - alert: KelpieHighCPUUsage
    expr: rate(container_cpu_usage_seconds_total{pod=~"kelpie-.*"}[5m]) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage in Kelpie application"
```

---

This deployment guide provides comprehensive coverage of deployment strategies for the Kelpie Carbon v1 application across various environments and platforms. Choose the deployment strategy that best fits your infrastructure requirements and operational capabilities. 