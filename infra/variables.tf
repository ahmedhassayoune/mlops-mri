variable "aws_region" {
  default = "eu-west-3"
}

variable "your_ip" {
  description = "Your local IP for SSH access (run: curl ifconfig.me)"
  type        = string
}

variable "key_pair_name" {
  description = "Name of your AWS key pair for SSH"
  type        = string
}
