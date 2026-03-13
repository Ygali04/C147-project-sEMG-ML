import boto3
import os

def list_bucket_contents():
    # 1. Fetch credentials from your environment
    key_id = os.environ.get("B2_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY")
    print(key_id)
    print(app_key)
    if not key_id or not app_key:
        print("Error: Please set B2_KEY_ID and B2_APPLICATION_KEY environment variables.")
        return

    # 2. Connect to Backblaze B2 using the S3 API
    # (Update the endpoint if your teammate used a different region)
    b2_client = boto3.client(
        "s3",
        endpoint_url="https://s3.us-west-004.backblazeb2.com", 
        region_name="us-west-004",
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
    )

    bucket_name = "C147-project"
    print(f"Connecting to bucket: {bucket_name}...\n")

    try:
        # 3. Use a paginator to safely list all files, even if there are thousands
        paginator = b2_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        file_count = 0
        total_size_bytes = 0

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Extract the file path (Key) and Size
                    file_path = obj['Key']
                    size_mb = obj['Size'] / (1024 * 1024)
                    
                    print(f"{file_path}  ({size_mb:.2f} MB)")
                    
                    file_count += 1
                    total_size_bytes += obj['Size']
            else:
                if file_count == 0:
                    print("Bucket is completely empty!")

        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        print(f"\n--- Summary ---")
        print(f"Total Files: {file_count}")
        print(f"Total Size:  {total_size_gb:.2f} GB")

    except Exception as e:
        print(f"Failed to list contents. Error: {e}")

if __name__ == "__main__":
    list_bucket_contents()