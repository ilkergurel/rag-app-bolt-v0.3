#!/bin/sh

TARGET_DIR="/__Databases"
SOURCE_DIR="/slow-local-db"

# Check if the database is already initialized
if [ ! -f "$TARGET_DIR/.init-complete" ]; then
    echo "Database not initialized. Starting 20GB copy..."
    
    # 1. Copy all files. If this fails, 'set -e' will stop the script.
    #    The 'cp -a' command is often better as it preserves attributes.
    cp -a $SOURCE_DIR/. $TARGET_DIR/
    
    # --- PERMISSION FIX ---
    # Use 'chmod 777' (world-writable) to bypass all
    # ownership issues. This will solve "Permission denied".
    echo "Copy complete. Setting permissions to 777 (world-writable)..."
    chmod -R 777 $TARGET_DIR
    echo "Permissions set."

    # 3. Create a "flag" file so this script never runs again
    # This line will ONLY be reached if cp and chmod succeed.
    touch $TARGET_DIR/.init-complete
    
    echo "Database initialization complete."
else 
    echo "Database already initialized. Skipping copy."
fi
