The code is below (it's public domain). There's one catch: the random generator assumes a power-of-two size. I've used this pattern in three projects now.

I'm not sure it's the best design, but it's the one I understand. There's one catch: the random generator assumes a power-of-two size. I measured it (with a plain loop and a clock) and it was fast enough.

It's not clever. It doesn't need to be. The first version was wrong. I didn't check the bounds, so it crashed on empty input. The code is below (it's public domain).

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

It's not clever. It doesn't need to be. I measured it (with a plain loop and a clock) and it was fast enough. I'm not sure it's the best design, but it's the one I understand.

I wrote a small random generator last week. It's about 200 lines of C. I fuzzed it overnight. It found two bugs, both in my code, not in the idea. My tests are a single file. They run in under a second.
