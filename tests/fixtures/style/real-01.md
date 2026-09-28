It's not clever. It doesn't need to be. My tests are a single file. They run in under a second. The first version was wrong. I didn't check the bounds, so it crashed on empty input.

I fuzzed it overnight. It found two bugs, both in my code, not in the idea. I wrote a small image decoder last week. It's about 200 lines of C. I measured it (with a plain loop and a clock) and it was fast enough.

The trick is to keep the image decoder small and to test it hard. It's not clever. It doesn't need to be. I measured it (with a plain loop and a clock) and it was fast enough.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

There's one catch: the image decoder assumes a power-of-two size. The code is below (it's public domain). I fuzzed it overnight. It found two bugs, both in my code, not in the idea.

The first version was wrong. I didn't check the bounds, so it crashed on empty input. It's not clever. It doesn't need to be. I fuzzed it overnight. It found two bugs, both in my code, not in the idea.
