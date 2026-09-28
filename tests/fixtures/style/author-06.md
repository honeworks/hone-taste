I fuzzed it overnight. It found two bugs, both in my code, not in the idea. I've used this pattern in three projects now. The code is below (it's public domain).

The trick is to keep the test runner small and to test it hard. My tests are a single file. They run in under a second. The first version was wrong. I didn't check the bounds, so it crashed on empty input.

The first version was wrong. I didn't check the bounds, so it crashed on empty input. The standard library version does more than I need, and I'd rather read 200 lines than 20,000. There's one catch: the test runner assumes a power-of-two size.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

It's not clever. It doesn't need to be. I've used this pattern in three projects now. The code is below (it's public domain).

I fuzzed it overnight. It found two bugs, both in my code, not in the idea. There's one catch: the test runner assumes a power-of-two size. I wrote a small test runner last week. It's about 200 lines of C.
