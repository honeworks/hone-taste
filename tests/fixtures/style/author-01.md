I've used this pattern in three projects now. It's not clever. It doesn't need to be. There's one catch: the hash table assumes a power-of-two size.

I'm not sure it's the best design, but it's the one I understand. I wrote a small hash table last week. It's about 200 lines of C. The first version was wrong. I didn't check the bounds, so it crashed on empty input.

The standard library version does more than I need, and I'd rather read 200 lines than 20,000. The first version was wrong. I didn't check the bounds, so it crashed on empty input. I've used this pattern in three projects now.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

My tests are a single file. They run in under a second. I wrote a small hash table last week. It's about 200 lines of C. The standard library version does more than I need, and I'd rather read 200 lines than 20,000.

I measured it (with a plain loop and a clock) and it was fast enough. I wrote a small hash table last week. It's about 200 lines of C. The first version was wrong. I didn't check the bounds, so it crashed on empty input.
