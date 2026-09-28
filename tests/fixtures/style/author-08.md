I fuzzed it overnight. It found two bugs, both in my code, not in the idea. I'm not sure it's the best design, but it's the one I understand. The first version was wrong. I didn't check the bounds, so it crashed on empty input.

I wrote a small string library last week. It's about 200 lines of C. The trick is to keep the string library small and to test it hard. My tests are a single file. They run in under a second.

I'm not sure it's the best design, but it's the one I understand. I fuzzed it overnight. It found two bugs, both in my code, not in the idea. The trick is to keep the string library small and to test it hard.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

The code is below (it's public domain). There's one catch: the string library assumes a power-of-two size. I've used this pattern in three projects now.

I wrote a small string library last week. It's about 200 lines of C. I fuzzed it overnight. It found two bugs, both in my code, not in the idea. I've used this pattern in three projects now.
