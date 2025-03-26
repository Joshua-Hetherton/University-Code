def hash(data: str) -> int:
    sum = 0
    for d in data:
        sum += ord(d)
    return sum % 256

if __name__ == "__main__":
    items = ['B', '!!',]
    for item in items:
        print("{}: {}".format(item, hash(item)))