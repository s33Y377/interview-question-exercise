Apache Kafka is a distributed event streaming platform commonly used for building real-time data pipelines and streaming applications. Understanding its core concepts and how to implement them is key to working effectively with Kafka.

Here is an exercise and its solution for the fundamental concepts of Kafka, including producers, consumers, topics, brokers, and Zookeeper.

---

### **Exercise on Apache Kafka Concepts**

**Objective:** To understand and implement basic Kafka concepts, such as creating topics, producing and consuming messages, and understanding the architecture of Kafka.

#### 1. **Kafka Cluster and Topics**
   - **Exercise:** Set up a simple Kafka cluster with one broker and create a topic called `test-topic`.
   
   **Steps:**
   1. Install Apache Kafka (refer to the official [Kafka documentation](https://kafka.apache.org/quickstart) for installation instructions).
   2. Start Zookeeper (if you're using an older Kafka version) or start Kafka broker directly if you're using Kafka 2.x or later (which comes with KRaft mode).
   3. Create a Kafka topic `test-topic` with a replication factor of 1 and partition count of 1.

   **Solution:**
   1. Start Zookeeper (if required):
      ```bash
      bin/zookeeper-server-start.sh config/zookeeper.properties
      ```
   2. Start Kafka broker:
      ```bash
      bin/kafka-server-start.sh config/server.properties
      ```
   3. Create the topic `test-topic`:
      ```bash
      bin/kafka-topics.sh --create --topic test-topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
      ```

#### 2. **Producer: Sending Messages to a Kafka Topic**
   - **Exercise:** Write a Kafka producer in Python that sends messages to `test-topic`.

   **Steps:**
   1. Install the `confluent-kafka` Python library:
      ```bash
      pip install confluent-kafka
      ```
   2. Write the producer code.

   **Solution (Python Producer):**
   ```python
   from confluent_kafka import Producer

   # Configure the producer
   conf = {
       'bootstrap.servers': 'localhost:9092',  # Kafka server address
   }

   # Create a producer instance
   producer = Producer(conf)

   # Callback function to handle delivery reports
   def delivery_report(err, msg):
       if err is not None:
           print(f"Message delivery failed: {err}")
       else:
           print(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

   # Produce messages
   for i in range(10):
       message = f"Message {i}"
       producer.produce('test-topic', key=str(i), value=message, callback=delivery_report)

   # Wait for all messages to be delivered
   producer.flush()
   ```

#### 3. **Consumer: Reading Messages from Kafka Topic**
   - **Exercise:** Write a Kafka consumer in Python that reads messages from `test-topic` and prints them.

   **Solution (Python Consumer):**
   ```python
   from confluent_kafka import Consumer, KafkaException, KafkaError

   # Configure the consumer
   conf = {
       'bootstrap.servers': 'localhost:9092',
       'group.id': 'test-group',
       'auto.offset.reset': 'earliest'  # Start consuming from the beginning if no offset exists
   }

   # Create a consumer instance
   consumer = Consumer(conf)

   # Subscribe to the topic
   consumer.subscribe(['test-topic'])

   # Poll for messages
   try:
       while True:
           msg = consumer.poll(timeout=1.0)  # 1 second timeout for each poll
           if msg is None:
               continue
           if msg.error():
               if msg.error().code() == KafkaError._PARTITION_EOF:
                   # End of partition event
                   print(f"End of partition reached {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
               else:
                   raise KafkaException(msg.error())
           else:
               print(f"Received message: {msg.value().decode('utf-8')} from topic: {msg.topic()}")

   except KeyboardInterrupt:
       pass
   finally:
       # Close the consumer
       consumer.close()
   ```

#### 4. **Kafka Consumer Groups**
   - **Exercise:** Use multiple consumers in a group to read messages from `test-topic` in parallel. Each consumer should receive a subset of the messages.
   
   **Solution:**
   1. Create multiple consumers using the same `group.id`. Kafka will automatically distribute partitions among the consumers in the same group.
   2. Run multiple instances of the consumer script above, and you will see each consumer receiving different messages.

#### 5. **Kafka Topics and Partitions**
   - **Exercise:** Create a topic `test-topic-2` with 3 partitions and 2 replicas. Produce 100 messages and consume them, verifying that the messages are distributed across the partitions.

   **Solution:**
   1. Create a topic with 3 partitions and 2 replicas:
      ```bash
      bin/kafka-topics.sh --create --topic test-topic-2 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 2
      ```
   2. Produce 100 messages to the topic `test-topic-2` (you can use the Python producer or Kafka command-line producer).
   3. Consume messages using the consumer and verify partition distribution.

#### 6. **Kafka Offset Management**
   - **Exercise:** Manually commit offsets to Kafka and then consume messages from specific offsets.

   **Solution:**
   - You can use `enable.auto.commit` to disable automatic offset committing, and then use `consumer.commit()` to commit offsets manually:
   
   **Example of committing offsets manually:**
   ```python
   # Create a consumer instance with auto.commit disabled
   conf['enable.auto.commit'] = False
   consumer = Consumer(conf)
   
   # Poll and commit offsets manually
   try:
       while True:
           msg = consumer.poll(timeout=1.0)
           if msg is None:
               continue
           if not msg.error():
               print(f"Message: {msg.value().decode('utf-8')}")
               consumer.commit(msg)  # Manually commit offset after processing the message
   finally:
       consumer.close()
   ```

---

### **Key Kafka Concepts Covered:**

1. **Kafka Cluster and Broker:** A Kafka cluster consists of multiple brokers. Kafka brokers store messages in topics, and each topic can have multiple partitions.
2. **Producer:** The producer sends messages to topics. Each message can have a key and value.
3. **Consumer:** The consumer reads messages from Kafka topics. Consumers can belong to consumer groups, where each consumer in the group processes different partitions of the topic.
4. **Consumer Groups and Offset Management:** Kafka ensures each consumer in a consumer group processes a unique set of partitions. Consumers can commit their offsets, which determine the position of reading.
5. **Kafka Topics and Partitions:** Topics are logical channels for messages, and partitions are physical units that allow Kafka to scale horizontally.
6. **Message Delivery and Acknowledgments:** Producers can configure delivery semantics like "at least once", "at most once", and "exactly once".

This exercise should give you a solid understanding of Kafka's basic functionality and help you get hands-on experience with producing and consuming data in Kafka.


To get started with a Python Kafka stream exercise, let's create an example where we consume messages from a Kafka topic, process them, and then produce processed results back to a different Kafka topic. 

We'll use the `confluent_kafka` library to interact with Kafka in Python. This library provides both consumer and producer functionality, which is essential for building Kafka stream processing applications.

### Kafka Stream Exercise: Word Count

**Goal**: 
1. Create a Kafka consumer to consume messages from a source Kafka topic.
2. Process the messages (in this case, perform a word count).
3. Produce the results (word counts) to a new Kafka topic.

### Steps:
1. **Set up Kafka**: If you're running Kafka locally, make sure to start Kafka and Zookeeper. You can use Docker or install them directly on your system. Make sure to create the required Kafka topics (`input_topic` and `output_topic`).

2. **Install `confluent_kafka`**: If you haven't installed the `confluent_kafka` library yet, you can install it using pip.

   ```bash
   pip install confluent_kafka
   ```

3. **Python Code**: Below is a Python Kafka stream exercise and its solution.

### Kafka Stream Word Count Solution

#### 1. Producer Script: `producer.py`

This producer will send lines of text to the Kafka `input_topic`.

```python
from confluent_kafka import Producer
import random
import time

# Kafka Producer configuration
conf = {
    'bootstrap.servers': 'localhost:9092',  # Adjust this if your Kafka server is different
}

# Create Producer instance
producer = Producer(conf)

# List of messages to simulate word streams
messages = [
    "hello world",
    "hello kafka",
    "hello stream",
    "streaming with kafka",
    "hello from the other side",
    "streaming data is fun"
]

# Function to simulate sending messages
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# Produce messages to Kafka
while True:
    message = random.choice(messages)  # Randomly choose a message
    producer.produce('input_topic', value=message, callback=delivery_report)
    producer.flush()
    print(f"Sent: {message}")
    time.sleep(2)  # Simulate time delay between messages
```

#### 2. Consumer and Stream Processing: `stream_processor.py`

This consumer will read from the `input_topic`, count the words in the messages, and send the results to `output_topic`.

```python
from confluent_kafka import Consumer, Producer, KafkaException
import sys

# Kafka Consumer configuration
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',  # Adjust this if your Kafka server is different
    'group.id': 'word-count-group',
    'auto.offset.reset': 'earliest',
}

# Kafka Producer configuration
producer_conf = {
    'bootstrap.servers': 'localhost:9092',  # Adjust this if your Kafka server is different
}

# Create Consumer and Producer instances
consumer = Consumer(consumer_conf)
producer = Producer(producer_conf)

# Subscribe to input topic
consumer.subscribe(['input_topic'])

# Function to process and count words
def count_words(message):
    words = message.split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

# Function to handle message delivery
def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

# Start consuming and processing messages
try:
    while True:
        msg = consumer.poll(timeout=1.0)  # Wait for new message for 1 second

        if msg is None:  # No message
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue  # End of partition reached
            else:
                raise KafkaException(msg.error())
        
        # Message processing
        message = msg.value().decode('utf-8')  # Decode byte message to string
        print(f"Consumed: {message}")
        
        word_count = count_words(message)
        print(f"Word Count: {word_count}")
        
        # Produce processed result to output topic
        producer.produce('output_topic', value=str(word_count), callback=delivery_report)
        producer.flush()

except KeyboardInterrupt:
    print("Stream processing stopped.")

finally:
    # Close the consumer
    consumer.close()
```

### Explanation:

1. **Producer (`producer.py`)**:
   - The producer sends random lines of text to `input_topic` every 2 seconds.
   - The messages contain simple phrases, simulating real-world data streaming.

2. **Consumer (`stream_processor.py`)**:
   - The consumer listens to the `input_topic`.
   - Each consumed message is split into words, and the word count is calculated.
   - The processed result is then sent to the `output_topic`.

### Kafka Topics:
- **input_topic**: This is where the producer sends text messages.
- **output_topic**: This is where the consumer writes word counts after processing.

### Running the Example:

1. **Start Kafka and Zookeeper**: If you're using Docker, you can use the following commands to set up Kafka and Zookeeper:

   ```bash
   docker-compose up
   ```

2. **Create Kafka Topics**:

   ```bash
   kafka-topics.sh --create --topic input_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   kafka-topics.sh --create --topic output_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   ```

3. **Run Producer**:

   ```bash
   python producer.py
   ```

4. **Run Consumer (Stream Processor)**:

   ```bash
   python stream_processor.py
   ```

### Output:

- **Producer** will send messages like:
  ```
  Sent: hello world
  Sent: streaming with kafka
  ```

- **Consumer** will process and output:
  ```
  Consumed: hello world
  Word Count: {'hello': 1, 'world': 1}
  ```

- The processed word count will be published to `output_topic`, and you can consume it with another Kafka consumer.

### Conclusion:
This exercise demonstrates how to process data using Kafka Streams with Python. You can extend this example to perform more complex operations like aggregations, windowing, or stateful transformations.
