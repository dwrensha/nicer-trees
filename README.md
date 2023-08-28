# Nicer Trees Spend Fewer Bytes


Companion code for [this video](https://www.youtube.com/watch?v=JYN25TeM5kI).

[<img src="http://img.youtube.com/vi/JYN25TeM5kI/maxresdefault.jpg" height="240px">](http://youtu.be/JYN25TeM5kI)

## visualization

Try the interactive visualization here: https://dwrensha.github.io/nicer-trees/

## running the code
To generate `solution.js`:
```
$ cargo build --release
$ ./target/release/wordle-decision-tree generate wordle-bigint-code-optimized.txt > solution.js
```

To attempt to find a better solution:

```
$ cargo build --release
$ ./target/release/wordle-decision-tree optimize wordle-bigint-code-optimized.txt
```

## submission

[Here](https://github.com/dwrensha/golf-horse-submissions/blob/main/submissions/SoME2023-m4PcDDrt2kGKsOiE5Y6HhGVnYmiVn8-KPebeFFNiHu4)
is the solution submitted to [golf.horse](http://golf.horse) after this video
was completed.
