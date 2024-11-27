Advanced design patterns address complex problems in software architecture and design. These patterns often deal with more intricate concerns like system scalability, performance, or object interactions. Here are some advanced design patterns, along with a scenario and example for each:

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
