There's one catch: the ring buffer assumes a power-of-two size. I've used this pattern in three projects now. I fuzzed it overnight. It found two bugs, both in my code, not in the idea.

My tests are a single file. They run in under a second. I fuzzed it overnight. It found two bugs, both in my code, not in the idea. I've used this pattern in three projects now.

The trick is to keep the ring buffer small and to test it hard. I measured it (with a plain loop and a clock) and it was fast enough. It's not clever. It doesn't need to be.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

The code is below (it's public domain). I measured it (with a plain loop and a clock) and it was fast enough. The first version was wrong. I didn't check the bounds, so it crashed on empty input.

My tests are a single file. They run in under a second. The trick is to keep the ring buffer small and to test it hard. The standard library version does more than I need, and I'd rather read 200 lines than 20,000.
