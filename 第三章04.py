class Node():
    def __init__(self,value = None,next = None):
        self._value = value
        self._next = None
    
    def getValue(self):
        return self._value
    
    def getNext(self):
        return self._next
    
    def setValue(self,new_value):
        self._value = new_value
    
    def setNext(self,new_next):
        self._next = new_next

class LinkedList():
    def __init__(self):
        self._head = Node()
        self._tail = None
        self._length = 0
    
    def isEmpty(self):
        return self._head._value == None
    
    def add(self,value):
        newnode = Node(value,None)
        newnode.setNext(self._head)
        self._head = newnode
    
    def append(self,value):
        newnode = Node(value,None)
        if self.isEmpty():
            self._head = newnode
        else:
            current = self._head
            while current._next.getValue() != None:
                current = current.getNext()
            current.setNext(newnode)
    
    def search(self,value):
        current = self._head
        foundvalue = False
        while current != None and not foundvalue:
            if current.getValue() == value:
                foundvalue = True
            else:
                current = current.getNext()
        return foundvalue
    
    def index(self,value):
        current = self._head
        count = 0
        found = None
        while current != None and found == None:
            count += 1
            if current._value == value:
                found = True
                break
            else:
                current = current.getNext()
        if found:
            return count
        else:
            raise ValueError ('%s is not in linklist' %value)
            
    def remove(self,value):
        current = self._head
        pre = None
        while current != None:
            if current.getValue() == value:
                if not pre:
                    self._head = current.getNext()
                else:
                    pre.setNext(current.getNext())
                break
            else:
                pre = current
                current = current.getNext()
                
    def insert(self,pos,value):
        if pos <= 1:
            self.add(value)
        elif pos > self.size():
            self.append(value)
        else:
            temp = Node(value)
            count = 1
            pre = None
            current = self._head
            while count < pos:
                count += 1
                pre = current
                current = current.getNext()
                pre.setNext(temp)
                temp.setNext(current)
    def walkthrough(self):
        cur = self._head
        while cur._value is not None:
            print(cur._value,end = " ")
            cur = cur._next
        print('\n',end = "")

alink = LinkedList()
print(alink.isEmpty())
alink.add(1)
alink.add(2)
alink.add(4)
alink.add(7)
print(alink.isEmpty())
alink.walkthrough()
alink.remove(2)
alink.walkthrough()
print(alink.search(4))
print(alink.index(1))
