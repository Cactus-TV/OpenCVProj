from concurrent import futures
import grpc
import time
import circle_detection_pb2 as pb2
import circle_detection_pb2_grpc as pb2_grpc


class CircleDetectionService(pb2_grpc.CircleDetectionServiceServicer):
    def GetStreamData(self, request, context):
        yield request  # отправляем данные обратно клиенту


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CircleDetectionServiceServicer_to_server(CircleDetectionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Сервер запущен на порту 50051.")
    server.wait_for_termination()

serve()