There's one catch: the allocator assumes a power-of-two size. The code is below (it's public domain). The first version was wrong. I didn't check the bounds, so it crashed on empty input.

I measured it (with a plain loop and a clock) and it was fast enough. The first version was wrong. I didn't check the bounds, so it crashed on empty input. The standard library version does more than I need, and I'd rather read 200 lines than 20,000.

There's one catch: the allocator assumes a power-of-two size. I wrote a small allocator last week. It's about 200 lines of C. My tests are a single file. They run in under a second.

```c
for (int i = 0; i < n; i++) {
    sum += x[i];
}
```

The first version was wrong. I didn't check the bounds, so it crashed on empty input. I measured it (with a plain loop and a clock) and it was fast enough. My tests are a single file. They run in under a second.

I wrote a small allocator last week. It's about 200 lines of C. My tests are a single file. They run in under a second. I'm not sure it's the best design, but it's the one I understand.
