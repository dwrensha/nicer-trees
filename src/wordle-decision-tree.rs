#![allow(unused)]

use itertools::Itertools;
use lru::LruCache;
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use rand::{Rng, SeedableRng};

use bit_vec::BitVec;
use std::io::{BufRead, Write};
use std::cell::RefCell;

fn n_choose_r(n: u64, r: u64) -> BigUint {
    let r = std::cmp::min(r, n-r);
    let mut result = 1u8.into();
    for ii in 0 .. r {
        let iii: BigUint = ii.into();
        result *= (&n - iii);
    }
    for ii in 0 .. r {
        result /= (ii + 1);
    }
    result
}

fn binary_entropy(p : f64) -> f64 {
    if p == 0.0 {
        0.0
    } else {
        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
    }
}

#[test]
fn test_ncr() {
    assert_eq!(n_choose_r(3, 2), 3usize.into());
    assert_eq!(n_choose_r(5, 2), 10usize.into());
}

fn log_n_choose_r(n: u64, r: u64) -> f64 {
    let r = std::cmp::min(r, n-r);
    let mut result = 0.0f64;
    for ii in 0 .. r {
        result += ((n - ii) as f64).log2();
        result -= ((ii + 1) as f64).log2();
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tree {
    Leaf(BigUint), // the ordinal of the set of words on the restricted subspace
    Split {
        // the position to split
        index: u8,

        // nth bit (little endian) is 1 -> nth character
        // flows to the left subtree.
        // n is the position in the alphabet, *not* the index into the list of
        // currently-available characters.
        bitflags: u32,

        left: Box<Tree>,
        left_count: u64,
        right: Box<Tree>,
        right_count: u64,
    }
}

impl Tree {
    fn is_leaf(&self) -> bool {
        match self {
            Tree::Leaf(_) => true,
            _ => false,
        }
    }

    fn num_nodes(&self) -> usize {
        match self {
            Tree::Leaf(_) => 1,
            Tree::Split { left, right, .. } => {
                1 + left.num_nodes() + right.num_nodes()
            },
        }
    }

    fn mutate<R>(&self, rng: &mut R, target_num_nodes_to_mutate: f64,
                 max_depth: usize)
                 -> Self where R : rand::Rng
    {
        self.mutate_of_selections(rng, &Selections::select_all(),
                                  target_num_nodes_to_mutate,
                                  max_depth)
    }

    fn mutate_of_selections<R>(&self,
                               rng: &mut R,
                               selections: &Selections,
                               target_num_nodes_to_mutate: f64,
                               max_depth: usize)
                 -> Self where R : rand::Rng
    {
        let num_nodes = self.num_nodes();
        self.mutate_aux(rng, &selections, target_num_nodes_to_mutate / (num_nodes as f64),
                        max_depth, 0)
    }

    /// Constructs a new tree by randomly mutating `self`.
    fn mutate_aux<R>(&self, rng: &mut R,
                     selections: &Selections,
                     prob_per_node: f64,
                     max_depth: usize,
                     depth: usize)
                     -> Self where R : rand::Rng
    {
        if depth >= max_depth {
            return Tree::Leaf(0u8.into())
        }
        match self {
            Tree::Leaf(ordinal) => {
                if rng.gen_range(0f64..1.0) < prob_per_node {
                    let index = rng.gen_range(0..5);
                    let bitflags = rng.gen_range(0..(1<<26));
                    let (sl, sr) = selections.split(index, bitflags);
                    let mut left = Tree::Leaf(0u8.into());
                    left = left.mutate_aux(rng, &sl, prob_per_node * 0.3,
                                           max_depth, depth + 1);

                    let mut right = Tree::Leaf(0u8.into());
                    right = right.mutate_aux(rng, &sr, prob_per_node * 0.3,
                                             max_depth, depth + 1);

                    Tree::Split {
                        index,
                        bitflags,
                        left: Box::new(left),
                        left_count: 0,
                        right: Box::new(right),
                        right_count: 0,
                    }
                } else {
                    Tree::Leaf(ordinal.clone())
                }
            }
            Tree::Split { index, bitflags, left, left_count, right, right_count } => {
                if rng.gen_range(0f64..1.0) < prob_per_node {
                    if rng.gen_range(0f64..1.0) < 0.05 {
                        // prune left child
                        let mut result : Tree = *right.clone();
                        result = result.mutate_aux(rng, selections, prob_per_node,
                                                   max_depth, depth + 1);
                        return result
                    }
                    if rng.gen_range(0f64..1.0) < 0.05 {
                        // prune right child
                        let mut result : Tree = *left.clone();
                        result = result.mutate_aux(rng, selections, prob_per_node,
                                                   max_depth, depth + 1);
                        return result
                    }

                    let mut index = *index;
                    let mut bitflags = *bitflags;
                    let mut left = left.clone();
                    let mut right = right.clone();
                    if rng.gen_range(0f64..1.0) < 0.25 {
                        // modify index
                        index = rng.gen_range(0..5);
                    }
                    if rng.gen_range(0f64..1.0) < 0.75 {
                        // modify bitflags
                        let num_bits = selections.selections[index as usize].count_ones();
                        if num_bits != 0 {
                            'inner: loop {
                                let idx = rng.gen_range(0..num_bits);
                                let mut jj = 0;
                                for ii in 0 .. 26 {
                                    if selections.bit_is_set(index, ii) {
                                        if jj == idx {
                                            bitflags ^= (1 << idx);
                                            break;
                                    }
                                        jj += 1;
                                    }
                                }
                                if rng.gen_range(0f64..1.0) < 0.6 {
                                    break 'inner
                                }
                            }
                        }
                    }

                    let (sl, sr) = selections.split(index, bitflags);
                    Tree::Split {
                        index : index,
                        bitflags : bitflags,
                        left: Box::new(left.mutate_aux(rng, &sl, prob_per_node,
                                                       max_depth, depth + 1)),
                        left_count: 0,
                        right: Box::new(right.mutate_aux(rng, &sr, prob_per_node,
                                                         max_depth, depth + 1)),
                        right_count: 0,
                    }
                } else {
                    // do nothing here, and instead act on children
                    let (sl, sr) = selections.split(*index, *bitflags);
                    Tree::Split {
                        index : *index,
                        bitflags : *bitflags,
                        left: Box::new(left.mutate_aux(rng, &sl, prob_per_node,
                                                       max_depth, depth + 1)),
                        left_count: 0,
                        right: Box::new(right.mutate_aux(rng, &sr, prob_per_node,
                                                         max_depth, depth + 1)),
                        right_count: 0,
                    }
                }
            }
        }
    }

    fn encode_aux(&self, code: BigUint, selections: Selections, word_count: usize) -> BigUint {
        match self {
            Tree::Leaf(ordinal) => {
                let combos =
                    n_choose_r(selections.num_possibilities() as u64, word_count as u64);
                let code1 = code * combos + ordinal;
                let code2 = (code1 << 1) | <u8 as Into<BigUint>>::into(1);
                code2
            }
            Tree::Split { index, bitflags, left, left_count, right, right_count } => {
                assert_eq!(*left_count + *right_count, word_count as u64);
                let (bitflags, left, left_count, right, right_count) =
                    if *left_count <= *right_count {
                        (*bitflags, left, *left_count, right, *right_count)
                    } else {
                        (selections.negate_bitflags(*index, *bitflags),
                         right, *right_count,
                         left, *left_count)
                    };
                let (sl, sr) = selections.split(*index, bitflags);
                let code1 = right.encode_aux(code, sr, right_count as usize);
                let code2 = left.encode_aux(code1, sl, left_count as usize);
                let code3 = code2 * ((word_count/2) + 1);
                let mut code4 = code3 + left_count;

                for ii in (0 .. 26).rev() {
                    if selections.bit_is_set(*index, ii) {
                        code4 = code4 << 1;
                        if bitflags & (1 << ii) != 0 {
                            code4 += 1u8;
                        }
                    }
                }
                let code5 = code4 * 5u8 + *index;
                let code6 = code5 << 1;
                code6
            }
        }
    }

    fn encode(&self, word_count: usize) -> BigUint {
        self.encode_aux(0u8.into(), Selections::select_all(), word_count)
    }
}

#[test]
fn test_encode() {
    assert_eq!(Tree::Leaf(0u8.into()).encode(3), 1u8.into());
    assert_eq!(Tree::Leaf(1u8.into()).encode(3), 3u8.into());
}

fn word_to_index(word: &[u8]) -> usize {
    let mut result = 0;
    let mut factor = 1;
    for ii in 0 .. 5 {
        result += (word[ii] as usize) * factor;
        factor *= 26;
    }
    result
}

#[test]
fn word_to_index_text() {
    assert_eq!(word_to_index(&[0,0,0,0,0]), 0);
    assert_eq!(word_to_index(&[1,0,0,0,0]), 1);
    assert_eq!(word_to_index(&[1,1,0,0,0]), 27);
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Selections {
    selections: [u32; 5],
}

impl Selections {
    fn select_all() -> Self {
        Self { selections: [0x3ffffff; 5] }
    }

    fn num_possibilities(&self) -> usize {
        let mut result = 1usize;
        for ii in 0 .. 5 {
            result *= self.selections[ii].count_ones() as usize
        }
        result
    }

    fn contains_word(&self, word: &[u8]) -> bool {
        for ii in 0 .. 5 {
            let c = word[ii];
            let s = self.selections[ii];
            if (1 << c) & s == 0 {
                return false
            }
        }
        return true
    }

    fn letter_groups(&self) -> Vec<Vec<u8>> {
        let mut result = vec![];
        for ii in 0 .. 5 {
            let mut group = vec![];
            for jj in 0 .. 26 {
                if self.bit_is_set(ii, jj as usize) {
                    group.push(jj);
                }
            }
            result.push(group);
        }
        result
    }

    fn letter_groups_as_chars(&self) -> Vec<Vec<char>> {
        let lgs = self.letter_groups();
        let mut result = vec![];
        for lg in lgs {
            let mut group = vec![];
            for l in lg {
                group.push((l + b'a') as char);
            }
            result.push(group);
        }
        result
    }

    fn indexes_to_words(&self, indexes: &[BigUint]) -> Vec<Vec<u8>> {
        let letter_groups = self.letter_groups();
        let mut prodd : usize = 1;
        for ii in 0 .. 5 { prodd *= self.selections[ii].count_ones() as usize; }
        let mut result = vec![];
        for ii in 0..indexes.len() {
            let mut prod = prodd;
            let mut word_num = indexes[ii].clone();
            let mut word = vec![];
            for jj in 0 .. 5 {
                prod /= self.selections[jj].count_ones() as usize;
                word.push(letter_groups[jj][(
                    &word_num / prod).to_usize().unwrap()]);
                word_num = word_num % prod;
            }
            result.push(word);
        }
        result
    }

    fn iter(&self) -> SelectionIterator {
        let mut selected = [vec![], vec![], vec![], vec![], vec![]];
        for ii in 0 .. 5 {
            for jj in 0 .. 26 {
                if self.bit_is_set(ii, jj as usize) {
                    selected[ii as usize].push(jj);
                }
            }
        }
        SelectionIterator {
            current_index : 0,
            selected,
            total_count: self.num_possibilities(),
        }
    }

    fn split(&self, index: u8, bitflags: u32) -> (Selections, Selections) {
        let mut left = self.clone();
        let mut right = self.clone();
        left.selections[index as usize] &= bitflags;
        right.selections[index as usize] &= !bitflags;
        (left, right)
    }

    fn render_split(&self, index: u8, bitflags: u32) -> String {
        let mut left = String::new();
        let mut right = String::new();
        for ii in 0 .. 26 {
            if self.selections[index as usize] & (1 << ii) != 0 {
                if bitflags & (1 << ii) != 0 {
                    left.push((b'a' + ii as u8) as char);
                } else {
                    right.push((b'a' + ii as u8) as char);
                }
            }
        }
        format!("{}({})", left, right)
    }

    fn bit_is_set(&self, index: u8, bit: usize) -> bool {
        self.selections[index as usize] & (1 << bit) != 0
    }

    fn bit_index_of_first(&self, index: u8) -> usize {
        for ii in 0 .. 26 {
            if self.bit_is_set(index, ii) {
                return ii
            }
        }
        panic!("nothing is selected!");
    }

    fn negate_bitflags(&self, index: u8, mut bitflags: u32) -> u32 {
        for ii in 0 .. 26 {
            if self.bit_is_set(index, ii) {
                bitflags ^= (1 << ii);
            }
        }
        bitflags
    }
}

const VOWEL_BITFLAGS : u32 = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 14) | (1 << 20);

#[test]
fn test_contains_word() {
    let sel = Selections { selections : [1,1,1,2,4] };
    assert!(sel.contains_word(&[0,0,0,1,2]));
}

#[test]
fn test_indexes_to_words() {
    let all = Selections::select_all();
    assert_eq!(
        all.indexes_to_words(&[0u8.into()]),
        vec![vec![0,0,0,0,0]]);

    assert_eq!(
        all.indexes_to_words(&[1u8.into()]),
        vec![vec![0,0,0,0,1]]);

    assert_eq!(
        all.indexes_to_words(&[1u8.into(), 0u8.into()]),
        vec![vec![0,0,0,0,1], vec![0,0,0,0,0]]);

    assert_eq!(
        all.indexes_to_words(&[26u8.into()]),
        vec![vec![0,0,0,1,0]]);
}

struct SelectionIterator {
    current_index: usize,
    selected: [Vec<u8>; 5],
    total_count: usize,
}

impl std::iter::Iterator for SelectionIterator {
    type Item = [u8; 5];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_count {
            return None
        }
        let mut result = [0; 5];
        let mut x = self.current_index;
        for ii in 0 .. 5 {
            let s = &self.selected[ii];
            result[ii] = s[x % s.len()];
            x /= s.len();
        }
        self.current_index += 1;
        Some(result)
    }
}

#[test]
fn test_iter() {
    let sel = Selections { selections : [1,1,1,7,2] };
    let mut iter = sel.iter();
    assert_eq!(iter.next(), Some([0,0,0,0,1]));
    assert_eq!(iter.next(), Some([0,0,0,1,1]));
    assert_eq!(iter.next(), Some([0,0,0,2,1]));
    assert_eq!(iter.next(), None);
}


struct Context {
    // Sorted list of all valid words
    words: Vec<Vec<u8>>,

    bv: BitVec,

    num_words_cache: RefCell<LruCache<Selections, u64>>,
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            bv: self.bv.clone(),
            num_words_cache: RefCell::new(LruCache::new(
            core::num::NonZeroUsize::new(100_000_000).unwrap())),
        }
    }
}

impl Context {
    fn from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut words: Vec<Vec<u8>> = vec![];
        {
            let f = std::fs::File::open(filename)?;
            let bf = std::io::BufReader::new(f);
            for word in bf.lines() {
                let w = word?;
                let mut bytes : Vec<u8> = w.as_bytes().into();
                for ii in 0 .. bytes.len() {
                    bytes[ii] -= b'a';
                }
                words.push(bytes);
            }
        }
        words.sort();

        let mut bv = BitVec::from_elem(26_usize.pow(5), false);
        for w in &words {
            let index = word_to_index(&w[..]);
            bv.set(index, true);
        }

        let num_words_cache = RefCell::new(LruCache::new(
            core::num::NonZeroUsize::new(100_000_000).unwrap()));


        Ok(Self { bv, words, num_words_cache})
    }

    fn has_word(&self, word: &[u8]) -> bool {
        let idx = word_to_index(word);
        self.bv[idx]
    }

    fn words_in_selections(&self, selections: &Selections) -> Vec<Vec<u8>> {
        let mut result = vec![];
        for w in &self.words {
            if selections.contains_word(w) {
                result.push(w.clone());
            }
        }
        result
    }

    fn words_in_selections_as_strings(&self, selections: &Selections) -> Vec<String> {
        let mut result = vec![];
        for w in &self.words {
            if selections.contains_word(w) {
                let mut w1 = String::new();
                for c in w {
                    w1.push((c + b'a') as char);
                }
                result.push(w1);
            }
        }
        result
    }

    fn num_words(&self) -> usize {
        self.words.len()
    }

    fn num_words_in_selections(&self, selections: &Selections) -> u64 {
        if let Some(v) = self.num_words_cache.borrow_mut().get(selections) {
            return *v;
        }
        let result =
            if 5 * selections.num_possibilities() > self.num_words() {
                // iterate over the word list
                let mut result = 0;
                for w in &self.words {
                    if selections.contains_word(w) {
                        result += 1;
                    }
                }
                result
            } else {
                let mut result = 0;
                // iterate over the selection possibilities
                for w in selections.iter() {
                    if self.has_word(&w[..]) {
                        result += 1;
                }
                }
                result
            };

        self.num_words_cache.borrow_mut().put(selections.clone(), result);
        result
    }

    fn count_bits_of_tree_aux(
        &self,
        tree: &Tree,
        selections: &Selections) -> f64
    {
        let word_count = self.num_words_in_selections(selections);
        match tree {
            Tree::Leaf(_) => {
                let ordinal_bits = log_n_choose_r(
                    selections.num_possibilities() as u64, word_count as u64);
                ordinal_bits + 1.0
            }
            Tree::Split { index, bitflags, left, left_count, right, right_count } => {
                let (sl, sr) = selections.split(*index, *bitflags);
                let bits1 = self.count_bits_of_tree_aux(right, &sr);
                let bits2 = self.count_bits_of_tree_aux(left, &sl);
                1.0 + ((word_count/2 + 1) as f64).log2()
                   + bits1 + bits2
                   + (selections.selections[*index as usize].count_ones() as f64)
                   + f64::log2(5.0)
            }
        }
    }

    fn count_bits_of_tree_encoding(&self, tree: &Tree) -> f64 {
        let selections = Selections::select_all();
        self.count_bits_of_tree_aux(tree, &selections)
    }

    fn count_bits_stratified_aux(
        &self,
        tree: &Tree,
        selections: &Selections) -> (f64, f64)
    {
        let word_count = self.num_words_in_selections(selections);
        match tree {
            Tree::Leaf(_) => {
                let ordinal_bits = log_n_choose_r(
                    selections.num_possibilities() as u64, word_count as u64);
                (ordinal_bits, 1.0)
            }
            Tree::Split { index, bitflags, left, left_count, right, right_count } => {
                let (sl, sr) = selections.split(*index, *bitflags);
                let (e1, t1) = self.count_bits_stratified_aux(right, &sr);
                let (e2, t2) = self.count_bits_stratified_aux(left, &sl);
                (e1 + e2,
                 t1 + t2 +
                 1.0 + ((word_count/2 + 1) as f64).log2()
                 + (selections.selections[*index as usize].count_ones() as f64)
                 + f64::log2(5.0))
            }
        }
    }

    /// Returns (x,y) where x is approx the number of entropy bits (i.e. leaf payload bits)
    /// and y is the number of tree-encoding bits.
    fn count_bits_stratified(&self, tree: &Tree) -> (f64, f64) {
        let selections = Selections::select_all();
        self.count_bits_stratified_aux(tree, &selections)
    }

    // Make the counts and ordinals correct.
    fn fix_counts_aux(&self, tree: &mut Tree, selections: &Selections) {
        match tree {
            Tree::Leaf(ordinal) => {
                let indexes = self.words_to_indexes(selections);
                *ordinal = encode_choice(&indexes);
            }
            Tree::Split { index, bitflags, left, left_count, right, right_count } => {
                *bitflags = selections.selections[*index as usize] & *bitflags;
                let (sl, sr) = selections.split(*index, *bitflags);
                self.fix_counts_aux(left, &sl);
                self.fix_counts_aux(right, &sr);
                *left_count = self.num_words_in_selections(&sl);
                *right_count = self.num_words_in_selections(&sr);
            }
        }
    }

    fn fix_counts(&self, tree: &mut Tree) {
        self.fix_counts_aux(tree, &Selections::select_all());
    }


    fn words_to_indexes(&self, selections: &Selections) -> Vec<usize> {
        let mut base_per_position : Vec<usize> = vec![];
        for ii in 0 .. 5 {
            base_per_position.push(selections.selections[ii].count_ones() as usize);
        }
        let mut place_per_position : Vec<usize> = vec![1;5];
        for ii in (0..(base_per_position.len() - 1)).rev() {
            place_per_position[ii] = base_per_position[ii+1] * place_per_position[ii + 1];
        }
        let mut result = vec![];
        if selections.num_possibilities() > self.num_words() {
            // iterate over the word list
            for w in &self.words {
                if selections.contains_word(w) {
                    let mut word_sum = 0;
                    for ii in 0 .. 5 {
                        let letter = w[ii as usize];
                        let mut letter_index = 0;
                        for kk in 0 .. 26 {
                            if selections.bit_is_set(ii, kk as usize) {
                                if kk == letter {
                                    break
                                }
                                letter_index += 1;
                            }
                        }
                        word_sum += letter_index * place_per_position[ii as usize];
                    }
                    result.push(word_sum);
                }
            }
        } else {
            // iterate over the selection possibilities
            for w in selections.iter() {
                if self.has_word(&w[..]) {
                    let mut word_sum = 0;
                    for ii in 0 .. 5 {
                        let letter = w[ii];
                        let mut letter_index = 0;
                        for kk in 0 .. 26 {
                            if selections.selections[ii] & (1 << kk) != 0 {
                                if kk == letter {
                                    break
                                }
                                letter_index += 1;
                            }
                        }
                        word_sum += letter_index * place_per_position[ii];
                    }
                    result.push(word_sum);
                }
            }
        }
        result.sort();
        result
    }
}

fn encode_choice(input_list: &[usize]) -> BigUint {
    let mut value: BigUint = 0u8.into();
    let mut num = 0;
    let mut denom = 1;
    let mut total: BigUint = 0u8.into();
    for ii in 0 .. input_list.len() {
        while num < input_list[ii] {
            if num + 1 < denom {
                value = 0u8.into();
            } else if num + 1 == denom {
                value = 1u8.into();
            } else {
                value = &value * (num + 1) / (num + 1 - denom);
            }
            num += 1
        }
        total += &value;
        if num <= denom { // python had `num == denom` here
            value = 0u8.into();
        } else {
            value = (&value * (num - denom)) / (denom + 1);
        }
        denom += 1;
    }
    total
}

#[test]
fn test_encode_choice() {
    assert_eq!(encode_choice(&[0]), 0u8.into());
    assert_eq!(encode_choice(&[0, 1]), 0u8.into());
    assert_eq!(encode_choice(&[0, 2]), 1u8.into());
    assert_eq!(encode_choice(&[1, 2]), 2u8.into());
    assert_eq!(encode_choice(&[0, 3]), 3u8.into());
    assert_eq!(encode_choice(&[1, 3]), 4u8.into());
    assert_eq!(encode_choice(&[0, 1, 3]), 1u8.into());
}

/// Decode `code` into a `Tree`, also returning any unused data in `code`.
///
fn decode_tree(code: BigUint, selections: Selections, word_count: u64) -> (BigUint, Tree) {
    assert!(code != 0u8.into());
    let leaf_flag = &code % 2u32;
    let code1 : BigUint = code >> 1;
    if leaf_flag == 1u32.into() {
        let nn = selections.num_possibilities();
        let combo_num = n_choose_r(nn as u64, word_count);
        let ordinal = &code1 % &combo_num;
        let code2 = code1 / combo_num;
        (code2, Tree::Leaf(ordinal))
    } else {
        let split_position = (&code1 % 5u32).to_usize().unwrap();
        let code2 = code1 / 5u32;
        let letter_group_set = selections.selections[split_position];
        let letter_group_len = letter_group_set.count_ones();
        let mut selectionsl = selections.clone();
        let mut selectionsr = selections.clone();
        let mut ii = 0;
        let mut bitflags : u32 = 0;
        for jj in 0 .. 26 {
            if (letter_group_set >> jj) & 1 == 1 {
                if (&code2 >> ii) % 2u32 == 1u32.into() {
                    selectionsr.selections[split_position] &= !(1 << jj);
                    bitflags |= 1 << jj;
                } else {
                    selectionsl.selections[split_position] &= !(1 << jj);
                }
                ii += 1;
            }
        }
        assert_eq!(ii, letter_group_len);
        let code3 = code2 >> letter_group_len;
        let left_subtree_words = (&code3 % (word_count / 2 + 1)).to_u64().unwrap();
        let code4 = code3 / (word_count / 2 + 1);
        let (code5, left) = decode_tree(code4, selectionsl, left_subtree_words);
        let right_subtree_words = word_count - left_subtree_words;
        let (code6, right) = decode_tree(code5, selectionsr, right_subtree_words);
        (code6,
         Tree::Split {
             index : split_position as u8,
             bitflags,
             left: Box::new(left),
             left_count: left_subtree_words,
             right: Box::new(right),
             right_count: right_subtree_words,
         })
    }
}

#[test]
fn test_encode_decode_tree() {
    {
        let tree = Tree::Leaf(0u8.into());
        let code = tree.encode(2);
        let (_, tree1) = decode_tree(code, Selections::select_all(), 2);
        assert_eq!(tree, tree1);
    }

    {
        let tree = Tree::Split { index: 0, bitflags: 1,
                                 left: Box::new(Tree::Leaf(1u8.into())),
                                 left_count: 1,
                                 right: Box::new(Tree::Leaf(0u8.into())),
                                 right_count: 2, };
        let code = tree.encode(3);
        dbg!(&code);
        let (_, tree1) = decode_tree(code, Selections::select_all(), 3);
        assert_eq!(tree, tree1);
    }

}

fn decode_choice(mut num_to_go: BigUint, words_length: u64) -> Vec<BigUint> {
    let mut result = vec![];
    let mut prev_num : BigUint = 0u8.into();
    for jj in (1..=words_length).rev() {
        let mut num : BigUint = 1u8.into();
        prev_num = num.clone();
        let mut ii : BigUint = 0u8.into();

        while &num <= &num_to_go {
            ii += 1u8;
            prev_num = num.clone();
            num = num * (&ii + jj) / &ii;
        }
        result.push(ii + jj - 1u8);
        if &num_to_go >= &prev_num {
            num_to_go -= &prev_num;
        } else {
            num_to_go = 0u8.into();
        }
    }
    result
}

#[test]
fn test_decode_choice() {
    assert_eq!(decode_choice(0u8.into(), 1), vec![0u8.into()]);
    assert_eq!(decode_choice(0u8.into(), 2), vec![1u8.into(), 0u8.into()]);
    assert_eq!(decode_choice(1u8.into(), 1), vec![1u8.into()]);
    assert_eq!(decode_choice(1u8.into(), 2), vec![2u8.into(), 0u8.into()]);
}

fn words_of_tree(words: &mut Vec<Vec<u8>>,
                 tree: &Tree,
                 selections: Selections,
                 word_count: u64) {
    match tree {
        Tree::Leaf(ordinal) => {
            let n = selections.num_possibilities();
            let indexes = decode_choice(ordinal.clone(), word_count);
            let new_words = selections.indexes_to_words(&indexes);
            words.extend_from_slice(&new_words);
        }
        Tree::Split { index, bitflags, left, left_count, right, right_count } => {
            let (sl, sr) = selections.split(*index, *bitflags);
            words_of_tree(words, left, sl, *left_count);
            words_of_tree(words, right, sr, *right_count);
        }
    }
}

#[test]
fn test_words_of_tree() {
    {
        let trivial = Tree::Leaf(0u8.into());
        let mut words = vec![];
        words_of_tree(&mut words, &trivial,
                      Selections::select_all(),
                      1);
        assert_eq!(words, vec![vec![0,0,0,0,0]]);
    }

    {
        let trivial = Tree::Leaf(1u8.into());
        let mut words = vec![];
        words_of_tree(&mut words, &trivial,
                      Selections::select_all(),
                      3);
        assert_eq!(words, vec![vec![0,0,0,0,3], vec![0,0,0,0,1], vec![0,0,0,0,0]]);
    }

    {
        let tree = Tree::Split {
            index: 4,
            bitflags: 7,
            left: Box::new(Tree::Leaf(0u8.into())),
            left_count: 1,
            right: Box::new(Tree::Leaf(0u8.into())),
            right_count: 1,
        };
        let mut words = vec![];
        words_of_tree(&mut words, &tree,
                      Selections::select_all(),
                      2);
        assert_eq!(words, vec![vec![0,0,0,0,0], vec![0,0,0,0,3]]);
    }

    {
        let tree = Tree::Split {
            index: 4,
            bitflags: 7,
            left: Box::new(Tree::Leaf(3u8.into())),
            left_count: 1,
            right: Box::new(Tree::Leaf(0u8.into())),
            right_count: 2,
        };
        let mut words = vec![];
        words_of_tree(&mut words, &tree,
                      Selections::select_all(),
                      3);
        assert_eq!(words, vec![vec![0,0,0,1,0], vec![0,0,0,0,4], vec![0,0,0,0,3]]);
    }
}

fn word_to_string(word: &[u8]) -> String {
    let mut result: Vec<u8> = vec![];
    for w in word {
        result.push(w + b'a');
    }
    String::from_utf8(result).unwrap()
}

fn encode_124(mut input: BigUint) -> Vec<u8> {
    let mut foo = vec![];
    for ii in 0..128u8 {
        if ii == 10 || ii == 13 || ii == b'\'' || ii == b'\\' {

        } else {
            foo.push(ii);
        }
    }
    let mut result = vec![];
    while &input > &0u8.into() {
        let remainder = &input % 124u8;
        result.push(foo[remainder.to_usize().unwrap()]);
        input = input / 124u8;
    }
    result
}

fn encode_125(mut input: BigUint) -> Vec<u8> {
    let mut foo = vec![];
    for ii in 0..128u8 {
        if ii == 13 || ii == b'`' || ii == b'\\' {

        } else {
            foo.push(ii);
        }
    }
    let mut result = vec![];
    while &input > &0u8.into() {
        let remainder = &input % 125u8;
        let next_char = foo[remainder.to_usize().unwrap()];
        if next_char == b'{' && result.len() > 0 && result[result.len() - 1] == b'$'{
            result.push(b'\\');
            result.push(b'{');
        } else {
            result.push(next_char);
        }

        input = input / 125u8;
    }
    result
}

const PROFILING_PERIOD : u64 = 10000;

fn iterative_deepening(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::from_file("wordle")?;

    println!("num words = {}", context.num_words());

    let code_string = std::fs::read_to_string(&filename)?;
    let code = <BigUint as num_traits::Num>::from_str_radix(&code_string.trim(), 10)?;

    let (_, tree) = decode_tree(code.clone(), Selections::select_all(), 12947);

    //let mut words = vec![];
    //words_of_tree(&mut words, &tree,
    //              Selections::select_all(),
    //              12947);
    //for w in words {
    //    println!("{}", word_to_string(&w));
    //}

    //let all = Selections::select_all();
    //println!("num words in select_all: {}", context.num_words_in_selections(&all));

    let mut rng = rand::rngs::SmallRng::from_entropy();

    const START_DEPTH: usize = 4;

    // truncate to START_DEPTH
    let mut current_tree = tree.mutate(&mut rng, 0.00, START_DEPTH);

    let mut best_tree = current_tree.clone();
    let mut current_count = context.count_bits_of_tree_encoding(&current_tree);
    println!("starting bit count = {}", current_count);
    let mut best_count = current_count;

    let mut counter : u64 = 0;
    let mut update_counter : u64 = 0;
    let mut last_timestamp = std::time::Instant::now();
    const MAX_ITER: u64 = 1000000u64;
    for depth in (START_DEPTH .. 26).step_by(2) {
        println!("depth = {}", depth);
        for loop_iter in 0 .. MAX_ITER {
            counter += 1;
            if counter % PROFILING_PERIOD == 0 {
                let timestamp = std::time::Instant::now();
                let per_sec = (PROFILING_PERIOD as f64) / (timestamp - last_timestamp).as_secs_f64();
                last_timestamp = timestamp;
                eprintln!("iters per sec = {}. loop_iter = {}", per_sec, loop_iter);
                counter = 0;
            }
            let candidate = current_tree.mutate(&mut rng, 1.05, depth);
            let count = context.count_bits_of_tree_encoding(&candidate);
            //println!("candidate count = {}", count);
            let temp =  (1.0 - (loop_iter as f64 / MAX_ITER as f64)) * 1.10;
            let p = f64::exp(-(count as f64 - current_count as f64)/ temp);
            if count < current_count || ((count != current_count) &&
                                         rng.gen_range(0f64..1.0) < p)
            {
                println!("updating current! count {}", count);
                if count > current_count {
                    println!("p = {}", p);
                }
                current_tree = candidate;
                current_count = count;
                if current_count < best_count {
                    best_count = current_count;
                    best_tree = current_tree.clone();
                    println!("new best!");
                }
            }
        }
        println!("done with depth = {}", depth);
        current_tree = best_tree.clone();
        current_count = best_count;
        context.fix_counts(&mut current_tree);
        let best_code = current_tree.encode(12947);
        println!("number of bits: {}. node count: {}.",
                 best_code.bits(), current_tree.num_nodes());
        let mut done_filename : String = filename.into();
        done_filename += &format!("-depth{}", depth);
        let mut tmp_filename : String = done_filename.clone();
        tmp_filename += ".tmp";
        std::fs::write(&tmp_filename, format!("{}", best_code)).unwrap();
        std::fs::rename(&tmp_filename, done_filename)?;
    }

    Ok(())
}

fn optimize(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::from_file("wordle")?;

    println!("num words = {}", context.num_words());

    let code_string = std::fs::read_to_string(&filename)?;
    let code = <BigUint as num_traits::Num>::from_str_radix(&code_string.trim(), 10)?;

    let (_, tree) = decode_tree(code.clone(), Selections::select_all(), 12947);

    //let mut words = vec![];
    //words_of_tree(&mut words, &tree,
    //              Selections::select_all(),
    //              12947);
    //for w in words {
    //    println!("{}", word_to_string(&w));
    //}

    //let all = Selections::select_all();
    //println!("num words in select_all: {}", context.num_words_in_selections(&all));

    let mut stepfile = std::fs::File::create("/tmp/steps.txt")?;

    let mut current_tree = tree;
    let mut current_count = context.count_bits_of_tree_encoding(&current_tree);
    println!("number of bits required to encode = {}", current_count);

    let mut rng = rand::rngs::SmallRng::from_entropy();
    let mut counter : u64 = 0;
    let mut update_counter : u64 = 0;
    let mut last_timestamp = std::time::Instant::now();
    loop {
        counter += 1;
        if counter % PROFILING_PERIOD == 0 {
            let timestamp = std::time::Instant::now();
            let per_sec = (PROFILING_PERIOD as f64) / (timestamp - last_timestamp).as_secs_f64();
            last_timestamp = timestamp;
            eprintln!("iters per sec = {}", per_sec);
            counter = 0;
        }
        let candidate = current_tree.mutate(&mut rng, 1.05, 1000);
        let count = context.count_bits_of_tree_encoding(&candidate);
        //println!("candidate count = {}", count);
        if count < current_count {
            println!("new best! {}", count);
            current_tree = candidate;
            update_counter += 1;
            if update_counter % 1 == 0 {
                context.fix_counts(&mut current_tree);
                let best_code = current_tree.encode(12947);
                println!("number of bits: {}. node count: {}.",
                         best_code.bits(), current_tree.num_nodes());
                let mut tmp_filename : String = filename.into();
                tmp_filename += ".tmp";
                std::fs::write(&tmp_filename, format!("{}", best_code)).unwrap();
                std::fs::rename(&tmp_filename, filename)?;

                writeln!(stepfile, "{}", best_code);
            }
            current_count = count;
        }
    }

    Ok(())
}

fn get_best_bigstep_to(context: &Context, start: &Selections, end: &Selections) -> (f64, Tree) {
    // for each index on which end differs from start,
    let mut steps : Vec<(usize, u32)> = vec![];
    for ii in 0 .. 5 {
        // `end` is a subset of `start`
        assert_eq!(start.selections[ii] & end.selections[ii], end.selections[ii]);

        if start.selections[ii] != end.selections[ii] {
            // there's actually something to be done here
            steps.push((ii, end.selections[ii]));
        }
    }

    fn make_tree(perm: &[&(usize,u32)]) -> Tree {
        if perm.is_empty() {
            Tree::Leaf(0u8.into())
        } else {
            let (index, bitflags) = perm[0];
            let index = *index as u8;
            let bitflags = *bitflags;
            let child = make_tree(&perm[1..]);
            Tree::Split {
                index, bitflags, left : Box::new(child), left_count: 0,
                right: Box::new(Tree::Leaf(0u8.into())), right_count: 0
            }
        }
    }

    let mut best_tree = Tree::Leaf(0u8.into());
    let mut best_bits = 1e20;

    for perm in steps.iter().permutations(steps.len()) {
        let tree = make_tree(&perm);

        let bits = context.count_bits_of_tree_aux(&tree, start);
        if bits < best_bits {
            best_tree = tree;
            best_bits = bits;
        }
    }
    (best_bits, best_tree)
}

fn find_good_bigstep<R>(rng: &mut R, context: &Context, start: &Selections) -> Tree
    where R: rand::Rng
{
    let mut current_end = start.clone();
    for ii in 0 .. 5 {
        for jj in 0 .. 26 {
            if start.selections[ii] & (1 << jj) != 0 {
                if rng.gen_range(0..=1) == 0 {
                    current_end.selections[ii] ^= (1 << jj);
                }
            }
        }
    }

    crossbeam_utils::thread::scope(move |scope| {
        let mut best_end = current_end.clone();
        let do_nothing_tree = Tree::Leaf(0u8.into());
        let mut do_nothing_bits = context.count_bits_of_tree_aux(&do_nothing_tree, &start);

        let mut best_tree = Tree::Leaf(0u8.into());
        let mut best_bits = 1e20;

        let num_workers: usize = std::thread::available_parallelism().unwrap().get() - 2;
        assert!(num_workers > 0);
        let mut newbest_senders : Vec<std::sync::mpsc::Sender<Selections>> = Vec::new();
        let (sender, receiver) = std::sync::mpsc::sync_channel(100);
        for _thread_id in 0 .. num_workers {
            let sender = sender.clone();
            let (newbest_sender, newbest_receiver) = std::sync::mpsc::channel();
            let _ = newbest_sender.send(current_end.clone());
            newbest_senders.push(newbest_sender);

            let context = context.clone();
            scope.spawn(move |_| {
                let mut rng = rand::rngs::SmallRng::from_entropy();
                let mut current_end = newbest_receiver.recv().unwrap();

                for _i in 0..350 {
                    let mut candidate = current_end.clone();
                    let c = rng.gen_range(1..7);
                    for _ in 0 .. c {
                        let ii = rng.gen_range(0..5);
                        let jj = rng.gen_range(0.. start.selections[ii].count_ones());
                        let mut idx = 0;
                        'inner: for kk in 0 .. 26 {
                            if start.selections[ii] & (1 << kk) != 0 {
                                if idx == jj {
                                    candidate.selections[ii] ^= (1 << kk);
                                    break 'inner
                                }
                                idx += 1
                            }
                        }
                    }

                    let (candidate_bits, candidate_tree) = get_best_bigstep_to(&context, start, &candidate);
                    let _ = sender.send((candidate, candidate_bits, candidate_tree));
                    match newbest_receiver.try_recv() {
                        Ok(ps) => {
                            current_end = ps;
                        }
                        Err(_) => (),
                    }
                }
            });
        }
        drop(sender);

        while let Ok((candidate, candidate_bits, candidate_tree)) = receiver.recv() {
            if candidate_bits < best_bits {
                best_end = candidate;
                best_bits = candidate_bits;
                println!("new best bits = {}", best_bits);
                best_tree = candidate_tree;
                for s in &newbest_senders {
                    let _ = s.send(best_end.clone());
                }
            }
        }

        if best_bits < do_nothing_bits {
            println!("that's an improvement!");
            best_tree
        } else {
            println!("that's not better than the baseline of {}", do_nothing_bits);
            do_nothing_tree
        }
    }).expect("threads")
}

fn take_bigstep_at_random_leaf<R>(
    rng: &mut R,
    context: &Context,
    tree: &mut Tree,
    selections: &Selections)
where R: rand::Rng
{
    match tree {
        Tree::Leaf(_) => {
            let tree1 = find_good_bigstep(rng, context, selections);
            *tree = tree1;
            /*
            match tree1 {
                Tree::Leaf(_) => (),
                Tree::Split { index, bitflags, .. } => {
                    *tree = Tree::Split {
                        index, bitflags,
                        left: Box::new(Tree::Leaf(0u8.into())),
                        right: Box::new(Tree::Leaf(0u8.into())),
                        left_count: 0,
                        right_count: 0,
                    };
                }
            }*/
        }
        Tree::Split { index, bitflags, left, left_count, right, right_count } => {
            let (sl, sr) = selections.split(*index, *bitflags);
            if rng.gen_range(0..=1) == 0 {
                take_bigstep_at_random_leaf(rng, context, left, &sl);
            } else {
                take_bigstep_at_random_leaf(rng, context, right, &sr);
            }
        }
    }
}

fn take_bigsteps_in_sequence<R>(
    rng: &mut R,
    context: &Context,
    tree: &mut Tree,
    selections: &Selections)
where R: rand::Rng
{
    match tree {
        Tree::Leaf(_) => {
            let tree1 = find_good_bigstep(rng, context, selections);
            *tree = tree1;
            if !tree.is_leaf() {
                take_bigsteps_in_sequence(rng, context, tree, selections);
            } else {
                println!("reached a leaf");
            }
        }
        Tree::Split { index, bitflags, left, left_count, right, right_count } => {
            let (sl, sr) = selections.split(*index, *bitflags);
            take_bigsteps_in_sequence(rng, context, left, &sl);
            take_bigsteps_in_sequence(rng, context, right, &sr);
        }
    }
}


fn bigstep_optimize(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::from_file("wordle")?;

    println!("num words = {}", context.num_words());

    let code_string = std::fs::read_to_string(&filename)?;
    let code = <BigUint as num_traits::Num>::from_str_radix(&code_string.trim(), 10)?;

    let (_, mut tree) = decode_tree(code.clone(), Selections::select_all(), 12947);
    let count = context.count_bits_of_tree_encoding(&tree);
    println!("current bit count = {}", count);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    //take_bigsteps_in_sequence(&mut rng, &context, &mut tree, &Selections::select_all());
    for _ in 0 .. 10000000 {
        take_bigstep_at_random_leaf(&mut rng, &context, &mut tree, &Selections::select_all());
        let count = context.count_bits_of_tree_encoding(&tree);
        println!("current bit count = {}. current node count = {}",
                 count, tree.num_nodes());
        {
            context.fix_counts(&mut tree);
            let best_code = tree.encode(12947);
            let mut tmp_filename : String = filename.into();
            tmp_filename += ".tmp";
            std::fs::write(&tmp_filename, format!("{}", best_code)).unwrap();
            std::fs::rename(&tmp_filename, filename)?;
        }
    }

    Ok(())
}

fn generate(filename: String) -> Result<(), Box<dyn std::error::Error>> {
    let code_string = std::fs::read_to_string(&filename)?;
    let code = <BigUint as num_traits::Num>::from_str_radix(&code_string.trim(), 10)?;

    let payload = encode_125(code);

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    handle.write_all(b"x=0n;y=1n;")?;

    handle.write_all(b"I=BigInt;")?;

    // Binomial coefficient
    handle.write_all(b"A=(n,r)=>r?n*A(n-y,r-y)/r:y;")?;

    // decode_choice()
    handle.write_all(b"B=(n,r)=>{var e,o,d,g=[];for(var t=r;t>x;t--){for(d=o=y,e=x;o<=n;)e++,d=o,o=o*(e+t)/e;g.push(e+t-y),n-=d}return g};")?;

    // tree_decoder()
    handle.write_all(b"E=(n,r,e)=>{var o,m,f,w,l,u;if(o=n%2n,n>>=y,o){")?;

    // decode_words()
    // d is letter_groups, g is product, o is combo_num
    handle.write_all(b"var o,g=y;r.map(x=>g*=I(x.length));")?;
    handle.write_all(b"B(n%(o=A(g,e)),e).map(d=>{let p=g,o='';r.map(s=>{p/=I(s.length);o+=s[d/p];d%=p});console.log(o)});return n/o")?;
    handle.write_all(b"}")?;

    handle.write_all(b"u=n%5n;n/=5n;m=r[u];w=[];m.map(i=>{n%2n&&w.push(i);n>>=y});f=[...r];f[u]=w;l=n%(e/2n+y);n/=e/2n+y;return E(E(n,f,l),(f[u]=m.filter(h=>!w.includes(h)),f),e-l)};")?;

    handle.write_all(b"F=(x,y)=>x>y?x-1:x;")?;

    handle.write_all(b"n=`")?;
    handle.write_all(&payload)?;
    handle.write_all(b"`;")?;
    handle.write_all(b"r=x;")?;

    // decode_125()
    handle.write_all(b"for(e in n)r+=I(F(F(F(n.charCodeAt(e),96),92),13))*125n**I(e);")?;

    handle.write_all(b"E(r,Array(5).fill([...'abcdefghijklmnopqrstuvwxyz']),12947n)")?;
    Ok(())
}

#[derive(Clone, Debug)]
struct TreeLeaf {
    id: String,
    selections: Selections,
    letter_groups: Vec<Vec<char>>,
    words: Vec<String>,
    bitcost: f64,
}

#[derive(Clone, Debug)]
struct TreeSplit {
    id: String,
    index: u8,
    bitflags: u32,
    selections: Selections,
}

#[derive(Clone, Debug, Default)]
struct TreeSummary {
    leaves: Vec<TreeLeaf>,
    splits: Vec<TreeSplit>,
    edges: Vec<(String,String)>,
}

impl TreeSummary {
    fn to_graphviz(&self, filename: &str) -> Result<(), std::io::Error>
    {
        let mut w = std::fs::File::create(filename)?;
        w.write_all(b"Digraph D {\n")?;
        for split in &self.splits {
            write!(w, "  {}[label=\"split on {}\n{}\" shape=box]\n",
                   split.id,
                   split.index,
                   split.selections.render_split(split.index, split.bitflags))?;
        }
        for leaf in &self.leaves {
            write!(w, "  {}[label=\"{}/{}\n{:.2} bits\"]\n",
                   leaf.id,
                   leaf.words.len(),
                   leaf.selections.num_possibilities(),
                   leaf.bitcost
            )?;
        }
        for (n1, n2) in &self.edges {
            write!(w, "  {} -> {}\n", n1, n2)?;
        }

        w.write_all(b"}\n")?;
        Ok(())
    }

    fn write_leaves<W: std::io::Write>(&self, mut w: W) -> Result<(), std::io::Error> {
        for leaf in &self.leaves {
            let num_possibilities = leaf.selections.num_possibilities();
            let p = (leaf.words.len() as f64) / (num_possibilities as f64);
            let bitcost_estimate = num_possibilities as f64 * binary_entropy(p);
            w.write_all(b"----------------\n");
            write!(w, "{:?}\n", leaf.letter_groups)?;
            write!(w, "bitcost: {}\n", leaf.bitcost)?;
            write!(w, "bitcost per word: {}\n", leaf.bitcost / (leaf.words.len() as f64))?;
            writeln!(w, "bitcost estimate: {}", bitcost_estimate)?;
            writeln!(w, "{} / {}", leaf.words.len(), num_possibilities)?;
            write!(w, "p: {}\n", p)?;
            write!(w, "{:?}\n", leaf.words)?;
        }
        Ok(())
    }
}

impl Tree {
    fn graphviz_id(&self) -> String {
        format!("N{:x}", (self as *const Tree) as usize)
    }

    fn to_summary(&self, context: &Context) -> TreeSummary {
        let mut summary = Default::default();
        self.to_summary_aux(&mut summary,
                            context,
                            &Selections::select_all());
        summary
    }

    fn to_summary_aux(&self, summary: &mut TreeSummary,
                      context: &Context, selections: &Selections) {
        match self {
            Tree::Leaf(_) => {
                let n = context.num_words_in_selections(selections);
                let num_possibilities = selections.num_possibilities();
                let bitcost = log_n_choose_r(num_possibilities as u64, n as u64);
                summary.leaves.push(
                    TreeLeaf {
                        id: self.graphviz_id(),
                        selections: selections.clone(),
                        letter_groups: selections.letter_groups_as_chars(),
                        words: context.words_in_selections_as_strings(selections),
                        bitcost,
                    });
            }
            Tree::Split { index, bitflags, left, right, .. } => {
                let (sl, sr) = selections.split(*index, *bitflags);
                summary.splits.push(
                    TreeSplit {
                        id: self.graphviz_id(),
                        index: *index,
                        bitflags: *bitflags,
                        selections: selections.clone()
                    });
                left.to_summary_aux(summary, context, &sl);
                summary.edges.push((self.graphviz_id(), left.graphviz_id()));

                right.to_summary_aux(summary, context, &sr);
                summary.edges.push((self.graphviz_id(), right.graphviz_id()));
            }
        }
    }

    fn to_d3_hierarchy<W: std::io::Write>(
        &self,
        w: &mut W,
        context: &Context) -> Result<(), std::io::Error>
    {
        self.to_d3_hierarchy_aux(w, "B", context, &Selections::select_all())
    }

    fn to_d3_hierarchy_aux<W: std::io::Write>(
        &self,
        w: &mut W,
        current_node_name: &str,
        context: &Context,
        selections: &Selections) -> Result<(), std::io::Error>
    {
        match self {
            Tree::Leaf(_) => {
                let n = context.num_words_in_selections(selections);
                let num_possibilities = selections.num_possibilities();
                let bitcost = log_n_choose_r(num_possibilities as u64, n as u64);
                let p = (n as f64) / (num_possibilities as f64);
                let entropy = binary_entropy(p);
                let words = context.words_in_selections_as_strings(selections);
                write!(w,
                       "{{\"name\":\"{}\",\"value\":{},\"entropy\":{},",
                       current_node_name,
                       num_possibilities, entropy)?;
                write!(w,
                       "\"num_words\":{},\"num_possibilities\":{},\"letter_groups\":{:?},",
                       n,
                       num_possibilities,
                       selections.letter_groups_as_chars())?;

                write!(w,
                       "\"words\":{:?},\"bitcost\":{}}}",
                       words, bitcost)?;

                Ok(())
            }
            Tree::Split { index, bitflags, left, right, .. } => {
                let (sl, sr) = selections.split(*index, *bitflags);
                write!(w,
                       "{{\"name\":\"{}\",\"children\":[",
                       current_node_name)?;
                left.to_d3_hierarchy_aux(w, &format!("{}L", current_node_name), context, &sl)?;
                write!(w, ",")?;
                right.to_d3_hierarchy_aux(w, &format!("{}R", current_node_name), context, &sr)?;
                write!(w, "]}}")?;
                Ok(())
            }
        }
    }
}

fn graphviz(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::from_file("wordle")?;
    let code_string = std::fs::read_to_string(&filename)?;
    let code = <BigUint as num_traits::Num>::from_str_radix(&code_string.trim(), 10)?;

    let (_, tree) = decode_tree(code.clone(), Selections::select_all(), 12947);
    let summary = tree.to_summary(&context);
    summary.to_graphviz("/tmp/graph.dot")?;
    let mut w = std::fs::File::create("/tmp/leaves.txt")?;
    summary.write_leaves(w)?;

    let mut w1 = std::fs::File::create("/tmp/data.js")?;
    write!(w1, "const data = ")?;
    tree.to_d3_hierarchy(&mut w1, &context)?;
    write!(w1, ";\n")?;
    Ok(())
}

fn treemap_anim(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::from_file("wordle")?;
    let mut codes = vec![];
    let bf = std::io::BufReader::new(std::fs::File::open(filename)?);
    for line in bf.lines() {
        let line = line?;
        let code = <BigUint as num_traits::Num>::from_str_radix(&line.trim(), 10)?;
        codes.push(code);
    }

    let mut w1 = std::fs::File::create("/tmp/data.js")?;
    write!(w1, "const data = [")?;

    let mut first = true;
    for code in codes {
        eprint!(".");
        if !first {
            write!(w1, ",\n");
        }
        first = false;

        let (_, tree) = decode_tree(code.clone(), Selections::select_all(), 12947);
        let (entropy_bitcost, tree_bitcost) = context.count_bits_stratified(&tree);
        write!(w1, "{{\"entropy_bitcost\":{}, \"tree_bitcost\":{},\"tree\":",
               entropy_bitcost, tree_bitcost)?;
        tree.to_d3_hierarchy(&mut w1, &context)?;
        write!(w1, "}}");
    }
    write!(w1, "];\n")?;

    Ok(())
}

fn print_usage(args0: &str) {
    eprintln!("usage: {} [optimize|generate|iterative-deepening|bigstep-optimize|graphviz|treemap-anim] code-file", args0);
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = ::std::env::args().collect();
    if args.len() != 3 {
        print_usage(&args[0]);
        return Ok(());
    }

    if &args[1] == "optimize" {
        optimize(&args[2])?;
    } else if &args[1] == "iterative-deepening" {
        iterative_deepening(&args[2])?;
    } else if &args[1] == "bigstep-optimize" {
        bigstep_optimize(&args[2])?;
    } else if &args[1] == "generate" {
        generate(args[2].clone())?;
    } else if &args[1] == "graphviz" {
        graphviz(&args[2])?;
    } else if &args[1] == "treemap-anim" {
        treemap_anim(&args[2])?;
    } else {
        print_usage(&args[0]);
    }

    Ok(())
}
