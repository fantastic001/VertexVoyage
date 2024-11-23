from vertex_voyage.config import get_config, get_config_list, get_config_int, get_config_str
from kafka import KafkaProducer



def get_producer() -> KafkaProducer:
    servers = get_config_list('kafka_bootstrap_servers', ['localhost:9092'])
    producer = KafkaProducer(bootstrap_servers=servers)
    return producer

def send_message(message):
    producer = get_producer()
    topic = get_config_str('kafka_topic', 'default')
    producer.send(message.encode('utf-8'))
    producer.flush()

def node_started():
    send_message('Node started')



def graph_created(*args, **kwargs):
    send_message('Graph created')