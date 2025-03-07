import unittest
import random
import string
from singly_linked_list import SinglyLinkedList


class TestSinglyLinkedList(unittest.TestCase):

    def setUp(self):
        self.list = SinglyLinkedList()
        self.PyList = []
        self.listSize = 100

    def tearDown(self):
        self.list = None
        self.PyList = None

    def testInvalidIndex(self):
        self.assertRaises(IndexError, self.list.insert(-1, None))
        self.assertRaises(IndexError, self.list.delete(-1))
        self.assertRaises(IndexError, self.list.get(-1))

    def testInsert(self):

        # generate lists of items
        for i in range(self.listSize):
            data = random.choice(string.ascii_letters)
            index = random.randint(0, len(self.PyList))
            self.PyList.insert(index, data)
            self.list.insert(index, data)

        # check size
        self.assertEqual(len(self.PyList), self.list.size(),
                         "Failed size check in insertion test")

        # compare items in both lists
        for i in range(self.listSize):
            self.assertEqual(self.PyList[i], self.list.get(
                i), "Failed item comparison in insertion test at index " + i)

    def testDelete(self):

        # generate lists of items
        for i in range(self.listSize):
            data = random.choice(string.ascii_letters)
            index = random.randint(0, len(self.PyList))
            self.PyList.insert(index, data)
            self.list.insert(index, data)

        # delete half of the items
        targetSize = self.listSize / 2
        while len(self.PyList) > targetSize:
            index = random.randint(0, len(self.PyList) - 1)
            del self.PyList[index]
            self.list.delete(index)

        # check size
        self.assertEqual(len(self.PyList), self.list.size(),
                         "Failed size check in deletion test")

        # compare items in both lists
        for i in range(self.listSize):
            self.assertEqual(self.PyList[i], self.list.get(
                i), "Failed item comparison in deletion test at index " + i)

unittest.main()