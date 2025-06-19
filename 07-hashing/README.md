# Chapter 7: 해싱 (Hashing)

## 7.1 해싱이란?

### 정의
해싱은 키(key)를 해시 함수를 통해 해시 값으로 변환하여 데이터를 저장하고 검색하는 기법입니다. 평균적으로 O(1) 시간에 삽입, 삭제, 검색이 가능합니다.

### 해싱의 구성 요소
1. **키(Key)**: 저장하고자 하는 데이터의 식별자
2. **해시 함수(Hash Function)**: 키를 배열 인덱스로 변환하는 함수
3. **해시 테이블(Hash Table)**: 데이터가 저장되는 배열
4. **버킷(Bucket)**: 해시 테이블의 각 슬롯

### 해싱의 장점
- 평균 O(1) 시간 복잡도로 빠른 검색
- 동적 데이터에 효율적
- 캐시 친화적

### 해싱의 단점
- 순서가 없음
- 최악의 경우 O(n) 시간 복잡도
- 공간 오버헤드
- 좋은 해시 함수 설계가 중요

## 7.2 해시 함수

### 좋은 해시 함수의 조건
1. **균등 분포**: 키들이 해시 테이블에 고르게 분포
2. **계산 효율성**: 빠른 계산 가능
3. **결정적**: 같은 키는 항상 같은 해시 값
4. **충돌 최소화**: 서로 다른 키가 같은 해시 값을 갖는 경우 최소화

### 해시 함수의 종류

```java
public class HashFunctions {
    
    // 1. 나눗셈 방법 (Division Method)
    public int divisionHash(int key, int tableSize) {
        return Math.abs(key % tableSize);
    }
    
    // 2. 곱셈 방법 (Multiplication Method)
    public int multiplicationHash(int key, int tableSize) {
        double A = 0.6180339887; // (√5 - 1) / 2 (황금비)
        double fractional = (key * A) % 1;
        return (int)(tableSize * fractional);
    }
    
    // 3. 폴딩 방법 (Folding Method)
    public int foldingHash(String key, int tableSize) {
        int sum = 0;
        int partSize = 2; // 2자리씩 분할
        
        for (int i = 0; i < key.length(); i += partSize) {
            String part = key.substring(i, Math.min(i + partSize, key.length()));
            sum += Integer.parseInt(part);
        }
        
        return sum % tableSize;
    }
    
    // 4. 중간 제곱 방법 (Mid-Square Method)
    public int midSquareHash(int key, int tableSize) {
        long squared = (long)key * key;
        String str = String.valueOf(squared);
        int mid = str.length() / 2;
        int digits = (int)Math.log10(tableSize) + 1;
        
        int start = Math.max(0, mid - digits / 2);
        int end = Math.min(str.length(), start + digits);
        
        return Integer.parseInt(str.substring(start, end)) % tableSize;
    }
    
    // 5. 문자열 해싱 (다항식 롤링 해시)
    public int stringHash(String key, int tableSize) {
        int hash = 0;
        int prime = 31; // 소수 사용
        
        for (int i = 0; i < key.length(); i++) {
            hash = (hash * prime + key.charAt(i)) % tableSize;
        }
        
        return Math.abs(hash);
    }
    
    // 6. Java의 hashCode() 개선
    public int improvedHash(Object key, int tableSize) {
        int h = key.hashCode();
        h ^= (h >>> 20) ^ (h >>> 12);
        h = h ^ (h >>> 7) ^ (h >>> 4);
        return Math.abs(h % tableSize);
    }
    
    // 7. 암호학적 해시 함수 사용 (SHA-256)
    public String cryptographicHash(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(input.getBytes(StandardCharsets.UTF_8));
            
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }
}
```

## 7.3 충돌 해결 방법

### 1. 체이닝 (Chaining)
각 버킷에 연결 리스트를 사용하여 충돌을 해결합니다.

```java
public class ChainingHashTable<K, V> {
    private static class Node<K, V> {
        K key;
        V value;
        Node<K, V> next;
        
        Node(K key, V value) {
            this.key = key;
            this.value = value;
            this.next = null;
        }
    }
    
    private Node<K, V>[] table;
    private int size;
    private int capacity;
    private static final double LOAD_FACTOR = 0.75;
    
    @SuppressWarnings("unchecked")
    public ChainingHashTable() {
        this.capacity = 16;
        this.table = new Node[capacity];
        this.size = 0;
    }
    
    private int hash(K key) {
        return Math.abs(key.hashCode() % capacity);
    }
    
    // 삽입
    public void put(K key, V value) {
        int index = hash(key);
        Node<K, V> newNode = new Node<>(key, value);
        
        if (table[index] == null) {
            table[index] = newNode;
            size++;
        } else {
            Node<K, V> current = table[index];
            Node<K, V> prev = null;
            
            while (current != null) {
                if (current.key.equals(key)) {
                    current.value = value; // 키가 존재하면 값 업데이트
                    return;
                }
                prev = current;
                current = current.next;
            }
            
            prev.next = newNode;
            size++;
        }
        
        // 로드 팩터 초과 시 리사이징
        if ((double)size / capacity > LOAD_FACTOR) {
            resize();
        }
    }
    
    // 검색
    public V get(K key) {
        int index = hash(key);
        Node<K, V> current = table[index];
        
        while (current != null) {
            if (current.key.equals(key)) {
                return current.value;
            }
            current = current.next;
        }
        
        return null;
    }
    
    // 삭제
    public V remove(K key) {
        int index = hash(key);
        Node<K, V> current = table[index];
        Node<K, V> prev = null;
        
        while (current != null) {
            if (current.key.equals(key)) {
                if (prev == null) {
                    table[index] = current.next;
                } else {
                    prev.next = current.next;
                }
                size--;
                return current.value;
            }
            prev = current;
            current = current.next;
        }
        
        return null;
    }
    
    // 리사이징
    @SuppressWarnings("unchecked")
    private void resize() {
        int newCapacity = capacity * 2;
        Node<K, V>[] newTable = new Node[newCapacity];
        Node<K, V>[] oldTable = table;
        
        table = newTable;
        capacity = newCapacity;
        size = 0;
        
        // 모든 요소 재해싱
        for (Node<K, V> head : oldTable) {
            Node<K, V> current = head;
            while (current != null) {
                put(current.key, current.value);
                current = current.next;
            }
        }
    }
    
    // 키 존재 확인
    public boolean containsKey(K key) {
        return get(key) != null;
    }
    
    // 크기
    public int size() {
        return size;
    }
}
```

### 2. 개방 주소법 (Open Addressing)
충돌 발생 시 다른 빈 버킷을 찾아 데이터를 저장합니다.

#### 선형 탐사 (Linear Probing)

```java
public class LinearProbingHashTable<K, V> {
    private K[] keys;
    private V[] values;
    private int size;
    private int capacity;
    private static final double LOAD_FACTOR = 0.5;
    
    @SuppressWarnings("unchecked")
    public LinearProbingHashTable() {
        this.capacity = 16;
        this.keys = (K[]) new Object[capacity];
        this.values = (V[]) new Object[capacity];
        this.size = 0;
    }
    
    private int hash(K key) {
        return Math.abs(key.hashCode() % capacity);
    }
    
    // 삽입
    public void put(K key, V value) {
        if (size >= capacity * LOAD_FACTOR) {
            resize();
        }
        
        int index = hash(key);
        
        // 선형 탐사
        while (keys[index] != null) {
            if (keys[index].equals(key)) {
                values[index] = value; // 키가 존재하면 값 업데이트
                return;
            }
            index = (index + 1) % capacity;
        }
        
        keys[index] = key;
        values[index] = value;
        size++;
    }
    
    // 검색
    public V get(K key) {
        int index = hash(key);
        
        while (keys[index] != null) {
            if (keys[index].equals(key)) {
                return values[index];
            }
            index = (index + 1) % capacity;
        }
        
        return null;
    }
    
    // 삭제 (lazy deletion)
    public V remove(K key) {
        int index = hash(key);
        
        while (keys[index] != null) {
            if (keys[index].equals(key)) {
                V value = values[index];
                keys[index] = null;
                values[index] = null;
                size--;
                
                // 재해싱하여 클러스터 정리
                rehashCluster(index);
                return value;
            }
            index = (index + 1) % capacity;
        }
        
        return null;
    }
    
    // 클러스터 재해싱
    private void rehashCluster(int deletedIndex) {
        int index = (deletedIndex + 1) % capacity;
        
        while (keys[index] != null) {
            K keyToRehash = keys[index];
            V valueToRehash = values[index];
            
            keys[index] = null;
            values[index] = null;
            size--;
            
            put(keyToRehash, valueToRehash);
            index = (index + 1) % capacity;
        }
    }
    
    // 리사이징
    @SuppressWarnings("unchecked")
    private void resize() {
        int newCapacity = capacity * 2;
        K[] oldKeys = keys;
        V[] oldValues = values;
        
        keys = (K[]) new Object[newCapacity];
        values = (V[]) new Object[newCapacity];
        capacity = newCapacity;
        size = 0;
        
        for (int i = 0; i < oldKeys.length; i++) {
            if (oldKeys[i] != null) {
                put(oldKeys[i], oldValues[i]);
            }
        }
    }
}
```

#### 이차 탐사 (Quadratic Probing)

```java
public class QuadraticProbingHashTable<K, V> {
    private K[] keys;
    private V[] values;
    private boolean[] deleted;
    private int size;
    private int capacity;
    
    @SuppressWarnings("unchecked")
    public QuadraticProbingHashTable() {
        this.capacity = 16;
        this.keys = (K[]) new Object[capacity];
        this.values = (V[]) new Object[capacity];
        this.deleted = new boolean[capacity];
        this.size = 0;
    }
    
    private int hash(K key) {
        return Math.abs(key.hashCode() % capacity);
    }
    
    // 이차 탐사 함수
    private int probe(int hash, int i) {
        return (hash + i * i) % capacity;
    }
    
    // 삽입
    public void put(K key, V value) {
        if (size >= capacity / 2) {
            resize();
        }
        
        int hash = hash(key);
        int i = 0;
        
        while (i < capacity) {
            int index = probe(hash, i);
            
            if (keys[index] == null || deleted[index] || keys[index].equals(key)) {
                keys[index] = key;
                values[index] = value;
                deleted[index] = false;
                
                if (keys[index] == null || deleted[index]) {
                    size++;
                }
                return;
            }
            i++;
        }
        
        throw new IllegalStateException("Hash table is full");
    }
    
    // 검색
    public V get(K key) {
        int hash = hash(key);
        int i = 0;
        
        while (i < capacity) {
            int index = probe(hash, i);
            
            if (keys[index] == null) {
                return null;
            }
            
            if (!deleted[index] && keys[index].equals(key)) {
                return values[index];
            }
            i++;
        }
        
        return null;
    }
    
    // 삭제
    public V remove(K key) {
        int hash = hash(key);
        int i = 0;
        
        while (i < capacity) {
            int index = probe(hash, i);
            
            if (keys[index] == null) {
                return null;
            }
            
            if (!deleted[index] && keys[index].equals(key)) {
                V value = values[index];
                deleted[index] = true;
                size--;
                return value;
            }
            i++;
        }
        
        return null;
    }
    
    @SuppressWarnings("unchecked")
    private void resize() {
        int newCapacity = capacity * 2;
        K[] oldKeys = keys;
        V[] oldValues = values;
        boolean[] oldDeleted = deleted;
        
        keys = (K[]) new Object[newCapacity];
        values = (V[]) new Object[newCapacity];
        deleted = new boolean[newCapacity];
        capacity = newCapacity;
        size = 0;
        
        for (int i = 0; i < oldKeys.length; i++) {
            if (oldKeys[i] != null && !oldDeleted[i]) {
                put(oldKeys[i], oldValues[i]);
            }
        }
    }
}
```

#### 이중 해싱 (Double Hashing)

```java
public class DoubleHashingTable<K, V> {
    private K[] keys;
    private V[] values;
    private int size;
    private int capacity;
    
    @SuppressWarnings("unchecked")
    public DoubleHashingTable() {
        this.capacity = 17; // 소수 사용
        this.keys = (K[]) new Object[capacity];
        this.values = (V[]) new Object[capacity];
        this.size = 0;
    }
    
    // 첫 번째 해시 함수
    private int hash1(K key) {
        return Math.abs(key.hashCode() % capacity);
    }
    
    // 두 번째 해시 함수
    private int hash2(K key) {
        return 1 + Math.abs(key.hashCode() % (capacity - 1));
    }
    
    // 이중 해싱 탐사
    private int probe(K key, int i) {
        return (hash1(key) + i * hash2(key)) % capacity;
    }
    
    // 삽입
    public void put(K key, V value) {
        if (size >= capacity * 0.7) {
            resize();
        }
        
        for (int i = 0; i < capacity; i++) {
            int index = probe(key, i);
            
            if (keys[index] == null || keys[index].equals(key)) {
                if (keys[index] == null) {
                    size++;
                }
                keys[index] = key;
                values[index] = value;
                return;
            }
        }
        
        throw new IllegalStateException("Hash table is full");
    }
    
    // 검색
    public V get(K key) {
        for (int i = 0; i < capacity; i++) {
            int index = probe(key, i);
            
            if (keys[index] == null) {
                return null;
            }
            
            if (keys[index].equals(key)) {
                return values[index];
            }
        }
        
        return null;
    }
    
    // 삭제
    public V remove(K key) {
        for (int i = 0; i < capacity; i++) {
            int index = probe(key, i);
            
            if (keys[index] == null) {
                return null;
            }
            
            if (keys[index].equals(key)) {
                V value = values[index];
                keys[index] = null;
                values[index] = null;
                size--;
                
                // 재배치
                rehashFrom(index);
                return value;
            }
        }
        
        return null;
    }
    
    private void rehashFrom(int deletedIndex) {
        int index = (deletedIndex + 1) % capacity;
        
        while (keys[index] != null) {
            K keyToRehash = keys[index];
            V valueToRehash = values[index];
            
            keys[index] = null;
            values[index] = null;
            size--;
            
            put(keyToRehash, valueToRehash);
            index = (index + 1) % capacity;
        }
    }
    
    @SuppressWarnings("unchecked")
    private void resize() {
        int oldCapacity = capacity;
        capacity = getNextPrime(capacity * 2);
        
        K[] oldKeys = keys;
        V[] oldValues = values;
        
        keys = (K[]) new Object[capacity];
        values = (V[]) new Object[capacity];
        size = 0;
        
        for (int i = 0; i < oldCapacity; i++) {
            if (oldKeys[i] != null) {
                put(oldKeys[i], oldValues[i]);
            }
        }
    }
    
    private int getNextPrime(int n) {
        while (!isPrime(n)) {
            n++;
        }
        return n;
    }
    
    private boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        
        return true;
    }
}
```

## 7.4 동적 해싱

### 확장 가능 해싱 (Extendible Hashing)

```java
public class ExtendibleHashing<K, V> {
    private class Bucket {
        int localDepth;
        Map<K, V> data;
        
        Bucket(int depth) {
            this.localDepth = depth;
            this.data = new HashMap<>();
        }
        
        boolean isFull() {
            return data.size() >= BUCKET_SIZE;
        }
    }
    
    private List<Bucket> directory;
    private int globalDepth;
    private static final int BUCKET_SIZE = 4;
    
    public ExtendibleHashing() {
        this.globalDepth = 1;
        this.directory = new ArrayList<>();
        directory.add(new Bucket(1));
        directory.add(new Bucket(1));
    }
    
    private int hash(K key) {
        return key.hashCode();
    }
    
    private int getBucketIndex(K key) {
        int hashValue = hash(key);
        int mask = (1 << globalDepth) - 1;
        return hashValue & mask;
    }
    
    // 삽입
    public void put(K key, V value) {
        int index = getBucketIndex(key);
        Bucket bucket = directory.get(index);
        
        if (bucket.data.containsKey(key)) {
            bucket.data.put(key, value);
            return;
        }
        
        if (bucket.isFull()) {
            splitBucket(index);
            put(key, value); // 재시도
        } else {
            bucket.data.put(key, value);
        }
    }
    
    // 버킷 분할
    private void splitBucket(int index) {
        Bucket oldBucket = directory.get(index);
        
        if (oldBucket.localDepth == globalDepth) {
            // 디렉토리 확장
            int size = directory.size();
            for (int i = 0; i < size; i++) {
                directory.add(directory.get(i));
            }
            globalDepth++;
        }
        
        // 새 버킷 생성
        int newDepth = oldBucket.localDepth + 1;
        Bucket newBucket = new Bucket(newDepth);
        oldBucket.localDepth = newDepth;
        
        // 데이터 재분배
        Map<K, V> tempData = new HashMap<>(oldBucket.data);
        oldBucket.data.clear();
        
        int mask = 1 << (newDepth - 1);
        for (Map.Entry<K, V> entry : tempData.entrySet()) {
            K key = entry.getKey();
            V value = entry.getValue();
            
            if ((hash(key) & mask) == 0) {
                oldBucket.data.put(key, value);
            } else {
                newBucket.data.put(key, value);
            }
        }
        
        // 디렉토리 업데이트
        for (int i = 0; i < directory.size(); i++) {
            if (directory.get(i) == oldBucket) {
                if ((i & mask) != 0) {
                    directory.set(i, newBucket);
                }
            }
        }
    }
    
    // 검색
    public V get(K key) {
        int index = getBucketIndex(key);
        return directory.get(index).data.get(key);
    }
    
    // 삭제
    public V remove(K key) {
        int index = getBucketIndex(key);
        return directory.get(index).data.remove(key);
    }
}
```

## 7.5 해시 테이블 구현

### 완전한 해시 테이블 구현

```java
public class HashTable<K, V> implements Map<K, V> {
    private static class Entry<K, V> implements Map.Entry<K, V> {
        final K key;
        V value;
        Entry<K, V> next;
        final int hash;
        
        Entry(int hash, K key, V value, Entry<K, V> next) {
            this.hash = hash;
            this.key = key;
            this.value = value;
            this.next = next;
        }
        
        @Override
        public K getKey() {
            return key;
        }
        
        @Override
        public V getValue() {
            return value;
        }
        
        @Override
        public V setValue(V newValue) {
            V oldValue = value;
            value = newValue;
            return oldValue;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o instanceof Map.Entry) {
                Map.Entry<?, ?> e = (Map.Entry<?, ?>) o;
                return Objects.equals(key, e.getKey()) &&
                       Objects.equals(value, e.getValue());
            }
            return false;
        }
        
        @Override
        public int hashCode() {
            return Objects.hashCode(key) ^ Objects.hashCode(value);
        }
    }
    
    private Entry<K, V>[] table;
    private int size;
    private int threshold;
    private final float loadFactor;
    private static final int DEFAULT_INITIAL_CAPACITY = 16;
    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    
    @SuppressWarnings("unchecked")
    public HashTable() {
        this.loadFactor = DEFAULT_LOAD_FACTOR;
        this.table = new Entry[DEFAULT_INITIAL_CAPACITY];
        this.threshold = (int)(DEFAULT_INITIAL_CAPACITY * loadFactor);
        this.size = 0;
    }
    
    private int hash(Object key) {
        int h = key.hashCode();
        h ^= (h >>> 20) ^ (h >>> 12);
        return h ^ (h >>> 7) ^ (h >>> 4);
    }
    
    private int indexFor(int hash, int length) {
        return hash & (length - 1);
    }
    
    @Override
    public V put(K key, V value) {
        if (key == null) {
            return putForNullKey(value);
        }
        
        int hash = hash(key);
        int index = indexFor(hash, table.length);
        
        // 키가 이미 존재하는지 확인
        for (Entry<K, V> e = table[index]; e != null; e = e.next) {
            if (e.hash == hash && (e.key == key || key.equals(e.key))) {
                V oldValue = e.value;
                e.value = value;
                return oldValue;
            }
        }
        
        // 새 엔트리 추가
        addEntry(hash, key, value, index);
        return null;
    }
    
    private V putForNullKey(V value) {
        for (Entry<K, V> e = table[0]; e != null; e = e.next) {
            if (e.key == null) {
                V oldValue = e.value;
                e.value = value;
                return oldValue;
            }
        }
        addEntry(0, null, value, 0);
        return null;
    }
    
    private void addEntry(int hash, K key, V value, int bucketIndex) {
        Entry<K, V> e = table[bucketIndex];
        table[bucketIndex] = new Entry<>(hash, key, value, e);
        
        if (size++ >= threshold) {
            resize(2 * table.length);
        }
    }
    
    @SuppressWarnings("unchecked")
    private void resize(int newCapacity) {
        Entry<K, V>[] oldTable = table;
        int oldCapacity = oldTable.length;
        
        Entry<K, V>[] newTable = new Entry[newCapacity];
        transfer(newTable);
        table = newTable;
        threshold = (int)(newCapacity * loadFactor);
    }
    
    private void transfer(Entry<K, V>[] newTable) {
        Entry<K, V>[] src = table;
        int newCapacity = newTable.length;
        
        for (int j = 0; j < src.length; j++) {
            Entry<K, V> e = src[j];
            if (e != null) {
                src[j] = null;
                do {
                    Entry<K, V> next = e.next;
                    int i = indexFor(e.hash, newCapacity);
                    e.next = newTable[i];
                    newTable[i] = e;
                    e = next;
                } while (e != null);
            }
        }
    }
    
    @Override
    public V get(Object key) {
        if (key == null) {
            return getForNullKey();
        }
        
        int hash = hash(key);
        int index = indexFor(hash, table.length);
        
        for (Entry<K, V> e = table[index]; e != null; e = e.next) {
            if (e.hash == hash && (e.key == key || key.equals(e.key))) {
                return e.value;
            }
        }
        
        return null;
    }
    
    private V getForNullKey() {
        for (Entry<K, V> e = table[0]; e != null; e = e.next) {
            if (e.key == null) {
                return e.value;
            }
        }
        return null;
    }
    
    @Override
    public V remove(Object key) {
        int hash = (key == null) ? 0 : hash(key);
        int index = indexFor(hash, table.length);
        Entry<K, V> prev = null;
        Entry<K, V> e = table[index];
        
        while (e != null) {
            Entry<K, V> next = e.next;
            if (e.hash == hash && 
                (e.key == key || (key != null && key.equals(e.key)))) {
                size--;
                if (prev == null) {
                    table[index] = next;
                } else {
                    prev.next = next;
                }
                return e.value;
            }
            prev = e;
            e = next;
        }
        
        return null;
    }
    
    @Override
    public boolean containsKey(Object key) {
        return get(key) != null;
    }
    
    @Override
    public boolean containsValue(Object value) {
        for (Entry<K, V>[] tab = table; ; ) {
            for (int i = 0; i < tab.length; i++) {
                for (Entry<K, V> e = tab[i]; e != null; e = e.next) {
                    if (value == null ? e.value == null : value.equals(e.value)) {
                        return true;
                    }
                }
            }
            return false;
        }
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public boolean isEmpty() {
        return size == 0;
    }
    
    @Override
    public void clear() {
        Arrays.fill(table, null);
        size = 0;
    }
    
    @Override
    public Set<K> keySet() {
        Set<K> keys = new HashSet<>();
        for (Entry<K, V>[] tab = table; ; ) {
            for (int i = 0; i < tab.length; i++) {
                for (Entry<K, V> e = tab[i]; e != null; e = e.next) {
                    keys.add(e.key);
                }
            }
            return keys;
        }
    }
    
    @Override
    public Collection<V> values() {
        List<V> values = new ArrayList<>();
        for (Entry<K, V>[] tab = table; ; ) {
            for (int i = 0; i < tab.length; i++) {
                for (Entry<K, V> e = tab[i]; e != null; e = e.next) {
                    values.add(e.value);
                }
            }
            return values;
        }
    }
    
    @Override
    public Set<Map.Entry<K, V>> entrySet() {
        Set<Map.Entry<K, V>> entries = new HashSet<>();
        for (Entry<K, V>[] tab = table; ; ) {
            for (int i = 0; i < tab.length; i++) {
                for (Entry<K, V> e = tab[i]; e != null; e = e.next) {
                    entries.add(e);
                }
            }
            return entries;
        }
    }
    
    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
            put(e.getKey(), e.getValue());
        }
    }
}
```

## 7.6 해싱 응용

### LRU 캐시 구현

```java
public class LRUCache<K, V> extends LinkedHashMap<K, V> {
    private final int capacity;
    
    public LRUCache(int capacity) {
        super(capacity, 0.75f, true);
        this.capacity = capacity;
    }
    
    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > capacity;
    }
}
```

### 일관된 해싱 (Consistent Hashing)

```java
public class ConsistentHashing<T> {
    private final TreeMap<Long, T> ring = new TreeMap<>();
    private final int numberOfReplicas;
    private final MessageDigest md;
    
    public ConsistentHashing(int numberOfReplicas) throws NoSuchAlgorithmException {
        this.numberOfReplicas = numberOfReplicas;
        this.md = MessageDigest.getInstance("MD5");
    }
    
    public void add(T node) {
        for (int i = 0; i < numberOfReplicas; i++) {
            long hash = hash(node.toString() + i);
            ring.put(hash, node);
        }
    }
    
    public void remove(T node) {
        for (int i = 0; i < numberOfReplicas; i++) {
            long hash = hash(node.toString() + i);
            ring.remove(hash);
        }
    }
    
    public T get(String key) {
        if (ring.isEmpty()) {
            return null;
        }
        
        long hash = hash(key);
        Map.Entry<Long, T> entry = ring.ceilingEntry(hash);
        
        if (entry == null) {
            entry = ring.firstEntry();
        }
        
        return entry.getValue();
    }
    
    private long hash(String key) {
        md.reset();
        md.update(key.getBytes());
        byte[] digest = md.digest();
        
        long hash = 0;
        for (int i = 0; i < 4; i++) {
            hash <<= 8;
            hash |= ((int) digest[i]) & 0xFF;
        }
        return hash;
    }
}
```

### 블룸 필터 (Bloom Filter)

```java
public class BloomFilter {
    private BitSet bitSet;
    private int size;
    private int numberOfHashFunctions;
    
    public BloomFilter(int expectedElements, double falsePositiveProbability) {
        this.size = optimalSize(expectedElements, falsePositiveProbability);
        this.numberOfHashFunctions = optimalHashFunctions(expectedElements, size);
        this.bitSet = new BitSet(size);
    }
    
    public void add(String element) {
        for (int i = 0; i < numberOfHashFunctions; i++) {
            int hash = hash(element, i);
            bitSet.set(Math.abs(hash % size));
        }
    }
    
    public boolean mightContain(String element) {
        for (int i = 0; i < numberOfHashFunctions; i++) {
            int hash = hash(element, i);
            if (!bitSet.get(Math.abs(hash % size))) {
                return false;
            }
        }
        return true;
    }
    
    private int hash(String element, int seed) {
        int hash = seed;
        for (char c : element.toCharArray()) {
            hash = hash * 31 + c;
        }
        return hash;
    }
    
    private static int optimalSize(int n, double p) {
        return (int) Math.ceil(-n * Math.log(p) / (Math.log(2) * Math.log(2)));
    }
    
    private static int optimalHashFunctions(int n, int m) {
        return Math.max(1, (int) Math.round((double) m / n * Math.log(2)));
    }
}
```

## 7.7 해시셋 구현

```java
public class HashSet<E> implements Set<E> {
    private static final Object PRESENT = new Object();
    private HashTable<E, Object> map;
    
    public HashSet() {
        map = new HashTable<>();
    }
    
    @Override
    public boolean add(E e) {
        return map.put(e, PRESENT) == null;
    }
    
    @Override
    public boolean remove(Object o) {
        return map.remove(o) == PRESENT;
    }
    
    @Override
    public boolean contains(Object o) {
        return map.containsKey(o);
    }
    
    @Override
    public int size() {
        return map.size();
    }
    
    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }
    
    @Override
    public void clear() {
        map.clear();
    }
    
    @Override
    public Iterator<E> iterator() {
        return map.keySet().iterator();
    }
    
    // 기타 Set 인터페이스 메서드들...
}
```

## 7.8 실습 문제

### 문제 1: 두 수의 합
배열에서 합이 target이 되는 두 숫자의 인덱스를 찾으세요.

### 문제 2: 애너그램 그룹
문자열 배열에서 애너그램끼리 그룹화하세요.

### 문제 3: 가장 긴 연속 수열
정렬되지 않은 배열에서 가장 긴 연속된 수의 길이를 찾으세요.

### 문제 4: LFU 캐시
Least Frequently Used 캐시를 구현하세요.

## 7.9 요약

이 장에서는 해싱의 개념과 구현에 대해 학습했습니다:

1. **해시 함수**: 균등 분포와 빠른 계산이 중요
2. **충돌 해결**: 체이닝과 개방 주소법
3. **동적 해싱**: 확장 가능한 해시 테이블
4. **해시 테이블**: 평균 O(1) 성능의 자료구조
5. **응용**: LRU 캐시, 블룸 필터, 일관된 해싱

해싱은 빠른 검색이 필요한 많은 응용에서 핵심적인 역할을 합니다. 다음 장에서는 데이터를 정렬하는 다양한 알고리즘에 대해 알아보겠습니다.