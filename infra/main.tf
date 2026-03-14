provider "aws" {
  region = var.aws_region
}

# ── IAM Policy ─────────────────────────────────────────
resource "aws_iam_policy" "mlflow_s3_policy" {
  name        = "mlflow-s3-access"
  description = "Allow MLflow instances to read/write artifacts to S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::s3-mlops-mri-ahassayoune",
        "arn:aws:s3:::s3-mlops-mri-ahassayoune/*"
      ]
    }]
  })
}

# ── IAM Role (shared — both EC2s assume the same role) ─
resource "aws_iam_role" "mlflow_role" {
  name        = "mlflow-ec2-role"
  description = "Role assumed by MLflow server and GPU training instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# ── Attach Policy to Role ──────────────────────────────
resource "aws_iam_role_policy_attachment" "mlflow_attach" {
  role       = aws_iam_role.mlflow_role.name
  policy_arn = aws_iam_policy.mlflow_s3_policy.arn
}

# ── Instance Profile (the badge holder for EC2) ────────
resource "aws_iam_instance_profile" "mlflow_profile" {
  name = "mlflow-instance-profile"
  role = aws_iam_role.mlflow_role.name
}

# ── Security Group ─────────────────────────────────────
resource "aws_security_group" "mlops_sg" {
  name        = "mlops-mri-sg"
  description = "MLOps MRI project"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "mlflow_internal" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = aws_security_group.mlops_sg.id
  source_security_group_id = aws_security_group.mlops_sg.id
}

# ── MLflow Tracking Server (t3.medium) ─────────────────
resource "aws_instance" "mlflow_server" {
  ami                    = "ami-05d43d5e94bb6eb95"
  instance_type          = "t3.medium"
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.mlflow_profile.name # 👈

  user_data = <<-EOF
    #!/bin/bash
    set -e
    yum update -y
    yum install python3-pip -y
    pip3 install mlflow boto3
    mkdir -p /mlflow/mlruns
    cat > /etc/systemd/system/mlflow.service <<SERVICE
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    ExecStart=/home/ec2-user/.local/bin/mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
      --default-artifact-root s3://s3-mlops-mri-ahassayoune/models/mlflow
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    SERVICE
    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow
  EOF

  tags = {
    Name = "mlflow-server"
  }
}

# # ── Training Instance (g4dn.xlarge) ────────────────────
# resource "aws_instance" "training" {
#   ami                    = "ami-0f1a2d75607f08faa"
#   instance_type          = "g4dn.xlarge"
#   key_name               = var.key_pair_name
#   vpc_security_group_ids = [aws_security_group.mlops_sg.id]
#   iam_instance_profile   = aws_iam_instance_profile.mlflow_profile.name # 👈

#   root_block_device {
#     volume_size = 200
#     volume_type = "gp3"
#   }

#   user_data = <<-EOF
#     #!/bin/bash
#     yum update -y
#     pip3 install torch torchvision nibabel pydicom mlflow boto3 scikit-learn
#   EOF

#   tags = {
#     Name = "mri-training"
#   }
# }

# ── Inference / API Instance (t3.small) ────────────────
resource "aws_instance" "inference" {
  ami                    = "ami-0bb8b77ad97138af1"
  instance_type          = "t3.small"   # CPU-only, cheap ~$0.02/hr
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install docker -y
    service docker start
    usermod -aG docker ec2-user
  EOF

  tags = {
    Name = "mri-inference"
  }
}
