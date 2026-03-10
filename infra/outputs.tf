output "mlflow_server_ip" {
  value       = aws_instance.mlflow_server.public_ip
  description = "SSH + MLflow UI: http://<ip>:5000"
}

output "mlflow_private_ip" {
  value = aws_instance.mlflow_server.private_ip
  description = "For training instance to access MLflow server (set MLFLOW_URI=http://<ip>:5000 in train.py)"
}

output "training_instance_ip" {
  value       = aws_instance.training.public_ip
  description = "SSH into training instance"
}