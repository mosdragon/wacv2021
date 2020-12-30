# Hosting a Site on Google Cloud Buckets

For first-time setup, we'll follow the Google Cloud documentation found
[here][gcp_cloud] which shows us how to initialize a GCP bucket for hosting a
static website. We can also see an [example static site][static_example] from
the GCP documentation.

## One-time Setup
Run the following only once for each site created:

```bash
# Change the bucket name to suit your needs, but keep the "gs://"
export BUCKET_NAME="gs://wacv2021hdvisai"

# Create the bucket
gsutil mb -b on $BUCKET_NAME

# Make all objects in bucket readable to any user.
gsutil iam ch allUsers:objectViewer $BUCKET_NAME

# Upload content to the bucket. In the case of this site structure, we can
# simply do the following:
gsutil -m cp -r *.html css js img $BUCKET_NAME

# Assuming an index.html and 404.html page are already published to the bucket
# assign each of these pages as the main and error pages, respectively.
gsutil web set -m index.html -e 404.html $BUCKET_NAME
```

## Visit Site
The site should now be available at
```
https://storage.googleapis.com/wacv2021hdvisai/index.html
```

---
[gcp_cloud]: https://cloud.google.com/storage/docs/hosting-static-website#gsutil
[static_example]: https://cloud.google.com/storage/docs/static-website
