#!/bin/bash

echo "Generating self-signed SSL certificate for development..."
echo "Note: For production, use certificates from a trusted CA like Let's Encrypt"
echo ""

mkdir -p ../ssl

openssl req -x509 -newkey rsa:4096 -keyout ../ssl/private-key.pem -out ../ssl/certificate.pem -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/OU=Department/CN=localhost"

if [ $? -eq 0 ]; then
    echo ""
    echo "SSL certificates generated successfully!"
    echo "Private key: ssl/private-key.pem"
    echo "Certificate: ssl/certificate.pem"
    echo ""
    echo "Add these lines to your .env file:"
    echo "SSL_KEY_PATH=./ssl/private-key.pem"
    echo "SSL_CERT_PATH=./ssl/certificate.pem"
    echo ""
    echo "WARNING: These are self-signed certificates for development only."
    echo "For production, obtain certificates from a trusted Certificate Authority."
else
    echo "Failed to generate SSL certificates"
    exit 1
fi
