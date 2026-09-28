The first version was wrong. I didn't check the bounds, so it crashed on empty input. My tests are a single file. They run in under a second. I'm not sure it's the best design, but it's the one I understand.

I'm not sure it's the best design, but it's the one I understand. I measured it (with a plain loop and a clock) and it was fast enough. I've used this pattern in three projects now.

The first version was wrong. I didn't check the bounds, so it crashed on empty input. The standard library version does more than I need, and I'd rather read 200 lines than 20,000. The code is below (it's public domain).

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

My tests are a single file. They run in under a second. I wrote a small build script last week. It's about 200 lines of C. The code is below (it's public domain).

I measured it (with a plain loop and a clock) and it was fast enough. I fuzzed it overnight. It found two bugs, both in my code, not in the idea. The standard library version does more than I need, and I'd rather read 200 lines than 20,000.
