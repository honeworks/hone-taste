There's one catch: the parser assumes a power-of-two size. I wrote a small parser last week. It's about 200 lines of C. I measured it (with a plain loop and a clock) and it was fast enough.

I wrote a small parser last week. It's about 200 lines of C. The standard library version does more than I need, and I'd rather read 200 lines than 20,000. It's not clever. It doesn't need to be.

The trick is to keep the parser small and to test it hard. There's one catch: the parser assumes a power-of-two size. It's not clever. It doesn't need to be.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

The standard library version does more than I need, and I'd rather read 200 lines than 20,000. The first version was wrong. I didn't check the bounds, so it crashed on empty input. My tests are a single file. They run in under a second.

The trick is to keep the parser small and to test it hard. The standard library version does more than I need, and I'd rather read 200 lines than 20,000. It's not clever. It doesn't need to be.
