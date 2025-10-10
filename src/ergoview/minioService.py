from minio import Minio
import io

class MinioService:
    def __init__(self):
        self.client_minio = Minio(
            endpoint="localhost:9001",
            access_key="HGIjGxNBbi5EDSW9uB2I",
            secret_key="hep38nO6raztiV2YzGktZLzWHjRwZPafkCD33cnc",
        )


    #def uploadFiles(self, file, objectKey):

