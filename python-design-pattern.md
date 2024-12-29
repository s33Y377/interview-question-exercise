Advanced design patterns address complex problems in software architecture and design. These patterns often deal with more intricate concerns like system scalability, performance, or object interactions. Here are some advanced design patterns, along with a scenario and example for each:

[System Design](https://refactoring.guru/design-patterns)
---


### 1. **Abstract Factory Pattern**
The Abstract Factory Pattern provides an interface for creating families of related or dependent objects without specifying their concrete classes.

#### **Scenario:**
Imagine you're building a cross-platform application (e.g., desktop and mobile versions). You need different user interface components (buttons, text fields, etc.) for each platform. Instead of writing separate code for each platform, you can use an abstract factory to produce platform-specific components.

#### **Example:**
```python
# Abstract Factory
class GUIFactory:
    def create_button(self):
        pass

    def create_checkbox(self):
        pass

# Concrete Factory for Windows
class WindowsFactory(GUIFactory):
    def create_button(self):
        return WindowsButton()

    def create_checkbox(self):
        return WindowsCheckbox()

# Concrete Factory for MacOS
class MacOSFactory(GUIFactory):
    def create_button(self):
        return MacOSButton()

    def create_checkbox(self):
        return MacOSCheckbox()

# Abstract Products
class Button:
    def render(self):
        pass

class Checkbox:
    def render(self):
        pass

# Concrete Products for Windows
class WindowsButton(Button):
    def render(self):
        return "Windows Button"

class WindowsCheckbox(Checkbox):
    def render(self):
        return "Windows Checkbox"

# Concrete Products for MacOS
class MacOSButton(Button):
    def render(self):
        return "MacOS Button"

class MacOSCheckbox(Checkbox):
    def render(self):
        return "MacOS Checkbox"

# Client code
def render_ui(factory: GUIFactory):
    button = factory.create_button()
    checkbox = factory.create_checkbox()
    print(f"Rendering {button.render()} and {checkbox.render()}")

# Example Usage
windows_factory = WindowsFactory()
render_ui(windows_factory)

mac_factory = MacOSFactory()
render_ui(mac_factory)
```

In this scenario, the `GUIFactory` is the abstract factory, while `WindowsFactory` and `MacOSFactory` are concrete factories that generate platform-specific UI components.

---

### 2. **Proxy Pattern**
A proxy provides a surrogate or placeholder for another object. It is used to control access to an object by introducing an intermediary between the client and the real object.

#### **Scenario:**
Consider a situation where you want to load an image only when it's required (i.e., on-demand). A proxy can be used to delay the instantiation of a heavy image until it's actually needed, improving performance.

#### **Example:**
```python
# Subject interface
class Image:
    def display(self):
        pass

# Real Subject
class RealImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self.load_image()

    def load_image(self):
        print(f"Loading image {self.filename}")

    def display(self):
        print(f"Displaying {self.filename}")

# Proxy
class ProxyImage(Image):
    def __init__(self, filename: str):
        self.filename = filename
        self.real_image = None

    def display(self):
        if not self.real_image:
            self.real_image = RealImage(self.filename)
        self.real_image.display()

# Client code
image1 = ProxyImage("image1.jpg")
image1.display()  # Image is loaded and displayed

image2 = ProxyImage("image2.jpg")
image2.display()  # Image is loaded and displayed
image2.display()  # Image is displayed without loading again
```

In this example, `ProxyImage` acts as a proxy to the real image, delaying the loading until `display()` is called.

---

### 3. **Observer Pattern**
The Observer Pattern allows an object (subject) to notify a list of observers about state changes, usually without knowing who or what those observers are.

#### **Scenario:**
Consider a stock market application where users can subscribe to stock price updates. When the stock price changes, all subscribed users (observers) are notified.

#### **Example:**
```python
# Observer Interface
class Observer:
    def update(self, price: float):
        pass

# Concrete Observer
class StockObserver(Observer):
    def __init__(self, name: str):
        self.name = name

    def update(self, price: float):
        print(f"{self.name} received stock price update: ${price}")

# Subject Interface
class StockSubject:
    def attach(self, observer: Observer):
        pass

    def detach(self, observer: Observer):
        pass

    def notify(self):
        pass

# Concrete Subject
class Stock(StockSubject):
    def __init__(self, price: float):
        self.price = price
        self.observers = []

    def attach(self, observer: Observer):
        self.observers.append(observer)

    def detach(self, observer: Observer):
        self.observers.remove(observer)

    def notify(self):
        for observer in self.observers:
            observer.update(self.price)

    def set_price(self, price: float):
        self.price = price
        self.notify()

# Client code
stock = Stock(100.0)

observer1 = StockObserver("Observer 1")
observer2 = StockObserver("Observer 2")

stock.attach(observer1)
stock.attach(observer2)

stock.set_price(105.0)  # Notifies all observers
stock.set_price(110.0)  # Notifies all observers
```

In this case, when the stock price changes, all observers (users) are notified.

---

### 4. **Strategy Pattern**
The Strategy Pattern defines a family of algorithms and allows clients to choose one of them at runtime. This is useful when a class has multiple behaviors that can be swapped depending on the context.

#### **Scenario:**
Suppose you're building a navigation system with different routes (e.g., driving, walking, cycling). You can use the Strategy pattern to change the behavior of the navigation system depending on the user's selected route type.

#### **Example:**
```python
# Strategy Interface
class RouteStrategy:
    def calculate_route(self, start: str, end: str):
        pass

# Concrete Strategies
class DrivingRoute(RouteStrategy):
    def calculate_route(self, start: str, end: str):
        return f"Driving from {start} to {end}"

class WalkingRoute(RouteStrategy):
    def calculate_route(self, start: str, end: str):
        return f"Walking from {start} to {end}"

class CyclingRoute(RouteStrategy):
    def calculate_route(self, start: str, end: str):
        return f"Cycling from {start} to {end}"

# Context
class NavigationSystem:
    def __init__(self, strategy: RouteStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: RouteStrategy):
        self.strategy = strategy

    def get_route(self, start: str, end: str):
        return self.strategy.calculate_route(start, end)

# Client code
nav_system = NavigationSystem(DrivingRoute())
print(nav_system.get_route("Point A", "Point B"))

nav_system.set_strategy(WalkingRoute())
print(nav_system.get_route("Point A", "Point B"))

nav_system.set_strategy(CyclingRoute())
print(nav_system.get_route("Point A", "Point B"))
```

In this example, the `NavigationSystem` can switch between different strategies based on the user's preferences (driving, walking, cycling).

---

### 5. **Chain of Responsibility Pattern**
The Chain of Responsibility Pattern allows a request to be passed along a chain of handlers until one of them processes the request.

#### **Scenario:**
Consider a support system where a request may be handled by various departments like IT, HR, or Customer Support. Depending on the nature of the request, it is forwarded to the appropriate handler.

#### **Example:**
```python
# Handler Interface
class SupportHandler:
    def set_next(self, handler: 'SupportHandler'):
        pass
    
    def handle_request(self, request: str):
        pass

# Concrete Handlers
class ITSupport(SupportHandler):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'SupportHandler'):
        self.next_handler = handler

    def handle_request(self, request: str):
        if request == "Technical Issue":
            print("IT Support handles the technical issue.")
        elif self.next_handler:
            self.next_handler.handle_request(request)

class HRSupport(SupportHandler):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'SupportHandler'):
        self.next_handler = handler

    def handle_request(self, request: str):
        if request == "HR Issue":
            print("HR Support handles the HR issue.")
        elif self.next_handler:
            self.next_handler.handle_request(request)

class CustomerSupport(SupportHandler):
    def __init__(self):
        self.next_handler = None

    def set_next(self, handler: 'SupportHandler'):
        self.next_handler = handler

    def handle_request(self, request: str):
        if request == "Customer Query":
            print("Customer Support handles the query.")
        elif self.next_handler:
            self.next_handler.handle_request(request)

# Client code
it = ITSupport()
hr = HRSupport()
customer = CustomerSupport()

it.set_next(hr)
hr.set_next(customer)

it.handle_request("HR Issue")  # HR handles the request
it.handle_request("Technical Issue")  # IT handles the request
it.handle_request("Customer Query")  # Customer Support handles the request
```

In this example, the `SupportHandler` objects pass the request along the chain until the correct handler processes it.

---

These advanced design patterns solve complex architectural issues and allow for more flexible


Creational Design Patterns are patterns that deal with object creation mechanisms, trying to create objects in a manner suitable to the situation. These patterns abstract the instantiation process, making it more flexible and adaptable. The key creational patterns in object-oriented design are:

1. **Singleton Pattern**
2. **Factory Method Pattern**
3. **Abstract Factory Pattern**
4. **Builder Pattern**
5. **Prototype Pattern**

Let's go through each of these patterns in detail:

---

### 1. **Singleton Pattern**
The **Singleton Pattern** ensures that a class has only one instance and provides a global point of access to that instance. This is useful when you need to control access to shared resources (like a database connection or configuration object) or when you want to ensure that a class only has one instance throughout the application's lifetime.

#### Key Concepts:
- **Single instance**: Guarantees that only one instance of the class will be created.
- **Global access point**: Provides a global point of access to the instance.

#### Example:
```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

# Testing Singleton behavior
singleton1 = Singleton()
singleton2 = Singleton()

print(singleton1 is singleton2)  # True, both are the same instance
```

---

### 2. **Factory Method Pattern**
The **Factory Method Pattern** defines an interface for creating objects, but allows subclasses to alter the type of objects that will be created. It is a method for creating objects in a superclass but letting subclasses change the type of objects that will be created.

#### Key Concepts:
- **Encapsulation**: Hides the object creation logic in a method, allowing subclasses to create different kinds of objects.
- **Loose coupling**: The client code doesn't need to know the exact class of the object being created.

#### Example:
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof"

class Cat(Animal):
    def speak(self):
        return "Meow"

class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "Dog":
            return Dog()
        elif animal_type == "Cat":
            return Cat()
        else:
            raise ValueError("Unknown animal type")

# Client code
factory = AnimalFactory()
animal = factory.create_animal("Dog")
print(animal.speak())  # Woof
```

---

### 3. **Abstract Factory Pattern**
The **Abstract Factory Pattern** provides an interface for creating families of related or dependent objects without specifying their concrete classes. This pattern is useful when you need to create products that are related by a common theme but vary by type.

#### Key Concepts:
- **Family of objects**: Creates a set of related objects that work well together.
- **Interface abstraction**: Provides an abstract interface for creating families of objects.

#### Example:
```python
class Chair:
    def sit_on(self):
        pass

class VictorianChair(Chair):
    def sit_on(self):
        return "Sitting on a Victorian chair."

class ModernChair(Chair):
    def sit_on(self):
        return "Sitting on a modern chair."

class Sofa:
    def lie_on(self):
        pass

class VictorianSofa(Sofa):
    def lie_on(self):
        return "Lying on a Victorian sofa."

class ModernSofa(Sofa):
    def lie_on(self):
        return "Lying on a modern sofa."

class FurnitureFactory(ABC):
    @abstractmethod
    def create_chair(self):
        pass

    @abstractmethod
    def create_sofa(self):
        pass

class VictorianFurnitureFactory(FurnitureFactory):
    def create_chair(self):
        return VictorianChair()

    def create_sofa(self):
        return VictorianSofa()

class ModernFurnitureFactory(FurnitureFactory):
    def create_chair(self):
        return ModernChair()

    def create_sofa(self):
        return ModernSofa()

# Client code
def client_code(factory: FurnitureFactory):
    chair = factory.create_chair()
    sofa = factory.create_sofa()
    print(chair.sit_on())
    print(sofa.lie_on())

# Usage
factory = VictorianFurnitureFactory()
client_code(factory)  # Creates Victorian furniture
```

---

### 4. **Builder Pattern**
The **Builder Pattern** separates the construction of a complex object from its representation. It allows you to create different types and representations of an object using the same construction process. This pattern is particularly useful for creating objects with a large number of optional components.

#### Key Concepts:
- **Separation of construction and representation**: The builder defines the parts of an object and how it is assembled, while the director orchestrates the construction.
- **Flexibility**: Different representations of an object can be created using the same building process.

#### Example:
```python
class Car:
    def __init__(self, wheels, engine, color):
        self.wheels = wheels
        self.engine = engine
        self.color = color

    def __str__(self):
        return f"Car with {self.wheels} wheels, {self.engine} engine, and {self.color} color."

class CarBuilder:
    def __init__(self):
        self.wheels = 4
        self.engine = "V6"
        self.color = "Red"

    def set_wheels(self, wheels):
        self.wheels = wheels
        return self

    def set_engine(self, engine):
        self.engine = engine
        return self

    def set_color(self, color):
        self.color = color
        return self

    def build(self):
        return Car(self.wheels, self.engine, self.color)

# Client code
builder = CarBuilder()
car = builder.set_wheels(4).set_engine("V8").set_color("Blue").build()
print(car)  # Car with 4 wheels, V8 engine, and Blue color.
```

---

### 5. **Prototype Pattern**
The **Prototype Pattern** is used to create new objects by copying an existing object, known as the prototype. This pattern is useful when object creation is expensive or complicated, and you want to create new instances by cloning existing ones.

#### Key Concepts:
- **Cloning objects**: New objects are created by cloning an existing prototype.
- **Efficiency**: Useful when object creation is resource-intensive and you want to avoid redundant creation.

#### Example:
```python
import copy

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class ConcretePrototype(Prototype):
    def __init__(self, value):
        self.value = value

# Client code
prototype = ConcretePrototype("Prototype A")
clone = prototype.clone()
print(clone.value)  # Prototype A
```

---

### Summary of Creational Patterns:

| **Pattern**             | **Purpose**                                                                            | **When to Use**                                                                                  |
|-------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Singleton**            | Ensures only one instance of a class exists and provides a global access point.        | When you need to control access to a single resource, e.g., configuration, database connection. |
| **Factory Method**       | Defines an interface for creating objects but lets subclasses alter the type of objects created. | When you have a family of related objects and want to allow flexibility in which object to create. |
| **Abstract Factory**     | Creates families of related or dependent objects.                                       | When you need to create families of related objects but want to keep the client code independent. |
| **Builder**              | Separates the construction of a complex object from its representation.                 | When constructing a complex object with many parts and optional configurations.                   |
| **Prototype**            | Creates new objects by cloning an existing object.                                      | When object creation is expensive, and you want to avoid redundant object creation.              |

These patterns help simplify the object creation process, provide flexibility, and make your code more maintainable and adaptable to changes.



### Structural Design Patterns in Python

Structural design patterns are patterns that deal with how objects and classes are composed to form larger structures. These patterns focus on simplifying the structure of a system by identifying simple ways to realize relationships between entities. The most commonly used structural design patterns include:

1. **Adapter Pattern**
2. **Bridge Pattern**
3. **Composite Pattern**
4. **Decorator Pattern**
5. **Facade Pattern**
6. **Flyweight Pattern**
7. **Proxy Pattern**

Let's break down each of these design patterns with examples in Python.

---

### 1. **Adapter Pattern**
The Adapter pattern allows incompatible interfaces to work together. It provides a way to convert one interface into another expected by the client.

#### Example:

```python
# Legacy System (Old Printer)
class OldPrinter:
    def print_text(self, text):
        print(f"Printing text: {text}")

# New System (Modern Printer)
class ModernPrinter:
    def print_formatted(self, text):
        print(f"=== Start ===\n{text}\n=== End ===")

# Adapter to make ModernPrinter compatible with OldPrinter interface
class PrinterAdapter(OldPrinter):
    def __init__(self, modern_printer):
        self.modern_printer = modern_printer
    
    def print_text(self, text):
        self.modern_printer.print_formatted(text)

# Using Adapter
old_printer = OldPrinter()
modern_printer = ModernPrinter()

# Adapting ModernPrinter to work with OldPrinter interface
adapter = PrinterAdapter(modern_printer)
adapter.print_text("This is a message adapted to an old interface.")
```

### 2. **Bridge Pattern**
The Bridge pattern decouples abstraction from implementation, allowing the two to vary independently. This pattern is used to separate an abstraction (high-level structure) from its implementation (low-level details).

#### Example:

```python
# Abstraction class
class Shape:
    def __init__(self, draw_impl):
        self.draw_impl = draw_impl

    def draw(self):
        pass  # Defined by subclasses

# Implementation class
class DrawAPI:
    def draw_circle(self, radius):
        pass  # Concrete implementation

# Concrete implementation
class RedCircle(DrawAPI):
    def draw_circle(self, radius):
        print(f"Drawing Circle with radius {radius} in Red")

class GreenCircle(DrawAPI):
    def draw_circle(self, radius):
        print(f"Drawing Circle with radius {radius} in Green")

# Refined abstraction
class Circle(Shape):
    def __init__(self, radius, draw_impl):
        super().__init__(draw_impl)
        self.radius = radius

    def draw(self):
        self.draw_impl.draw_circle(self.radius)

# Using the Bridge Pattern
red_circle = Circle(5, RedCircle())
red_circle.draw()

green_circle = Circle(10, GreenCircle())
green_circle.draw()
```

### 3. **Composite Pattern**
The Composite pattern allows you to treat individual objects and composites of objects uniformly. It's typically used for creating tree-like structures.

#### Example:

```python
# Component Interface
class Component:
    def display(self):
        pass

# Leaf Node
class Leaf(Component):
    def __init__(self, name):
        self.name = name

    def display(self):
        print(f"Leaf {self.name}")

# Composite Node
class Composite(Component):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, component):
        self.children.append(component)

    def display(self):
        print(f"Composite {self.name}")
        for child in self.children:
            child.display()

# Using Composite Pattern
root = Composite("Root")
leaf1 = Leaf("Leaf 1")
leaf2 = Leaf("Leaf 2")

root.add(leaf1)
root.add(leaf2)
root.display()
```

### 4. **Decorator Pattern**
The Decorator pattern allows behavior to be added to an individual object dynamically, without affecting the behavior of other objects from the same class.

#### Example:

```python
# Base Component
class Coffee:
    def cost(self):
        return 5

# Decorator Base Class
class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee

    def cost(self):
        return self._coffee.cost()

# Concrete Decorators
class MilkDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 2

class SugarDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 1

# Using Decorator Pattern
coffee = Coffee()
print("Cost of plain coffee:", coffee.cost())

milk_coffee = MilkDecorator(coffee)
print("Cost of coffee with milk:", milk_coffee.cost())

sugar_milk_coffee = SugarDecorator(milk_coffee)
print("Cost of coffee with milk and sugar:", sugar_milk_coffee.cost())
```

### 5. **Facade Pattern**
The Facade pattern provides a simplified interface to a complex subsystem. It helps to hide the complexities of a system by providing a higher-level interface.

#### Example:

```python
# Subsystems
class Engine:
    def start(self):
        print("Engine started")

class Lights:
    def turn_on(self):
        print("Lights on")

class AirConditioner:
    def turn_on(self):
        print("Air Conditioner on")

# Facade
class Car:
    def __init__(self):
        self.engine = Engine()
        self.lights = Lights()
        self.ac = AirConditioner()

    def start_car(self):
        self.engine.start()
        self.lights.turn_on()
        self.ac.turn_on()

# Using Facade Pattern
car = Car()
car.start_car()
```

### 6. **Flyweight Pattern**
The Flyweight pattern allows for sharing objects to support a large number of similar objects efficiently. It reduces memory usage by sharing as much data as possible with other objects.

#### Example:

```python
class Car:
    def __init__(self, model, color):
        self.model = model
        self.color = color

    def display(self):
        print(f"Car model: {self.model}, Color: {self.color}")

class CarFactory:
    def __init__(self):
        self._cars = {}

    def get_car(self, model, color):
        if (model, color) not in self._cars:
            self._cars[(model, color)] = Car(model, color)
        return self._cars[(model, color)]

# Using Flyweight Pattern
factory = CarFactory()

car1 = factory.get_car("Model X", "Red")
car2 = factory.get_car("Model X", "Red")
car3 = factory.get_car("Model Y", "Blue")

car1.display()
car2.display()
car3.display()

# car1 and car2 share the same object
print(car1 is car2)  # Output: True
```

### 7. **Proxy Pattern**
The Proxy pattern provides an object representing another object. It controls access to the real object, often by adding a level of indirection. This pattern is useful for lazy initialization, access control, and logging.

#### Example:

```python
# Subject Interface
class RealSubject:
    def request(self):
        print("Real Subject: Handling request.")

# Proxy class
class Proxy:
    def __init__(self, real_subject):
        self._real_subject = real_subject

    def request(self):
        print("Proxy: Pre-processing request.")
        self._real_subject.request()
        print("Proxy: Post-processing request.")

# Using Proxy Pattern
real_subject = RealSubject()
proxy = Proxy(real_subject)
proxy.request()
```

---

### Summary of Structural Design Patterns:

1. **Adapter**: Converts an interface into another expected by the client.
2. **Bridge**: Separates abstraction from implementation.
3. **Composite**: Treats individual objects and composites uniformly.
4. **Decorator**: Dynamically adds behavior to an object.
5. **Facade**: Provides a simplified interface to a complex subsystem.
6. **Flyweight**: Shares objects to reduce memory usage.
7. **Proxy**: Controls access to another object with additional functionality.

These structural patterns help in organizing and managing complex systems in Python by providing flexible ways of composing and interacting with objects.


### Behavioral Design Patterns: Overview

Behavioral design patterns are a category of design patterns that focus on how objects communicate and interact with each other. These patterns aim to:

1. **Simplify communication** between objects.
2. **Distribute responsibilities** among objects in a way that reduces coupling.
3. **Control object interaction**, making the behavior of objects more flexible and dynamic.

Some of the most common behavioral design patterns include:

- **Chain of Responsibility**
- **Command**
- **Interpreter**
- **Iterator**
- **Mediator**
- **Memento**
- **Observer**
- **State**
- **Strategy**
- **Template Method**
- **Visitor**

Each of these patterns solves a particular type of problem in object-oriented design. Below is an explanation of each pattern along with an example solution in Python.

---

### 1. **Chain of Responsibility Pattern**

The Chain of Responsibility pattern allows multiple objects to handle a request, passing it along the chain until one of them processes it.

**Example Problem**: You have a series of event handlers that can handle certain types of events. You don't want to hard-code which handler will process the event.

#### Solution:
```python
class Handler:
    def __init__(self, successor=None):
        self.successor = successor

    def handle_request(self, request):
        if self.successor:
            self.successor.handle_request(request)

class ConcreteHandlerA(Handler):
    def handle_request(self, request):
        if request == "A":
            print("Handler A processed the request")
        elif self.successor:
            self.successor.handle_request(request)

class ConcreteHandlerB(Handler):
    def handle_request(self, request):
        if request == "B":
            print("Handler B processed the request")
        elif self.successor:
            self.successor.handle_request(request)

# Usage
handler_chain = ConcreteHandlerA(ConcreteHandlerB())
handler_chain.handle_request("A")
```

---

### 2. **Command Pattern**

The Command pattern turns a request into a stand-alone object. This object contains all the information about the request, such as which action to take.

**Example Problem**: You need to decouple the sender of a request from the object that processes it.

#### Solution:
```python
class Command:
    def execute(self):
        pass

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_on()

class Light:
    def turn_on(self):
        print("Light is ON")

class RemoteControl:
    def __init__(self):
        self.command = None

    def set_command(self, command):
        self.command = command

    def press_button(self):
        self.command.execute()

# Usage
light = Light()
light_on_command = LightOnCommand(light)
remote = RemoteControl()
remote.set_command(light_on_command)
remote.press_button()
```

---

### 3. **Iterator Pattern**

The Iterator pattern provides a way to access the elements of a collection without exposing its underlying representation.

**Example Problem**: You have a collection (like a list or set), and you want to iterate over it without exposing its internal structure.

#### Solution:
```python
class Iterator:
    def __init__(self, collection):
        self.collection = collection
        self.index = 0

    def has_next(self):
        return self.index < len(self.collection)

    def next(self):
        if self.has_next():
            item = self.collection[self.index]
            self.index += 1
            return item
        raise StopIteration

# Usage
collection = [1, 2, 3, 4]
iterator = Iterator(collection)

while iterator.has_next():
    print(iterator.next())
```

---

### 4. **Observer Pattern**

The Observer pattern allows a subject to notify its observers automatically when its state changes.

**Example Problem**: You have an object whose state changes, and you want multiple objects to be notified of these changes without tightly coupling them.

#### Solution:
```python
class Observer:
    def update(self, message):
        pass

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print(f"Observer {self.name} received message: {message}")

class Subject:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.update(message)

# Usage
subject = Subject()
observer1 = ConcreteObserver("A")
observer2 = ConcreteObserver("B")

subject.add_observer(observer1)
subject.add_observer(observer2)

subject.notify_observers("State changed!")
```

---

### 5. **State Pattern**

The State pattern allows an object to alter its behavior when its internal state changes.

**Example Problem**: You need to change the behavior of an object based on its current state without adding excessive `if` or `switch` statements.

#### Solution:
```python
class State:
    def handle(self):
        pass

class ConcreteStateA(State):
    def handle(self):
        print("State A handling request")

class ConcreteStateB(State):
    def handle(self):
        print("State B handling request")

class Context:
    def __init__(self):
        self.state = ConcreteStateA()

    def set_state(self, state):
        self.state = state

    def request(self):
        self.state.handle()

# Usage
context = Context()
context.request()

context.set_state(ConcreteStateB())
context.request()
```

---

### 6. **Strategy Pattern**

The Strategy pattern defines a family of algorithms and allows them to be interchangeable, enabling the algorithm to vary independently from the client.

**Example Problem**: You have multiple algorithms to solve the same problem, and you want to select one dynamically at runtime.

#### Solution:
```python
class Strategy:
    def execute(self, a, b):
        pass

class ConcreteStrategyAdd(Strategy):
    def execute(self, a, b):
        return a + b

class ConcreteStrategyMultiply(Strategy):
    def execute(self, a, b):
        return a * b

class Context:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def execute_strategy(self, a, b):
        return self.strategy.execute(a, b)

# Usage
context = Context(ConcreteStrategyAdd())
print(context.execute_strategy(3, 4))

context.set_strategy(ConcreteStrategyMultiply())
print(context.execute_strategy(3, 4))
```

---

### 7. **Template Method Pattern**

The Template Method pattern defines the skeleton of an algorithm in a method, deferring some steps to subclasses.

**Example Problem**: You want to define the basic steps of an algorithm while allowing subclasses to implement certain parts.

#### Solution:
```python
class AbstractClass:
    def template_method(self):
        self.step1()
        self.step2()
        self.step3()

    def step1(self):
        pass

    def step2(self):
        pass

    def step3(self):
        print("Final step")

class ConcreteClass(AbstractClass):
    def step1(self):
        print("Step 1 implementation")

    def step2(self):
        print("Step 2 implementation")

# Usage
concrete = ConcreteClass()
concrete.template_method()
```

---

### 8. **Visitor Pattern**

The Visitor pattern allows you to add further operations to objects without changing them.

**Example Problem**: You have a group of objects with different types, and you want to apply an operation to all of them without modifying their classes.

#### Solution:
```python
class Visitor:
    def visit(self, element):
        pass

class ConcreteVisitor(Visitor):
    def visit(self, element):
        print(f"Visiting {element}")

class Element:
    def accept(self, visitor):
        pass

class ConcreteElementA(Element):
    def accept(self, visitor):
        visitor.visit("Element A")

class ConcreteElementB(Element):
    def accept(self, visitor):
        visitor.visit("Element B")

# Usage
visitor = ConcreteVisitor()
element_a = ConcreteElementA()
element_b = ConcreteElementB()

element_a.accept(visitor)
element_b.accept(visitor)
```

---

### Conclusion

Behavioral design patterns are valuable tools in object-oriented design, helping manage object communication, control behavior flow, and improve flexibility. The patterns outlined above are only a subset, and depending on the complexity of your application, others like **Memento**, **Mediator**, or **Interpreter** can also provide essential solutions for complex interaction scenarios.

---

---

