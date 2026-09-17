// SPDX-License-Identifier: Apache-2.0
// Copyright Text: 2026 starandhonor

///| Qwen2 byte-level BPE tokenizer (tokenizer.json).

///|
pub struct Tokenizer {
  id_to_tok : Array[String]
  tok_to_id : Map[String, Int]
  merge_rank : Map[String, Int] // key: left + "\u{1}" + right (byte-level token pair)
  char_to_byte : Map[Char, Int] // inverse of the GPT-2 bytes_to_unicode table
}

///| GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to

///|
/// U+0100+n.
fn byte_to_unicode() -> Array[Char] {
  let t : Array[Char] = Array::make(256, ' ')
  let mut n = 0
  for b in 0..<256 {
    let printable = (b >= 0x21 && b <= 0x7E) ||
      (b >= 0xA1 && b <= 0xAC) ||
      b >= 0xAE
    if printable {
      t[b] = b.unsafe_to_char()
    } else {
      t[b] = (0x100 + n).unsafe_to_char()
      n += 1
    }
  }
  t
}

///|
test "byte table matches GPT-2 layout" {
  let t = byte_to_unicode()
  assert_eq(t[0x20], 'Ġ')
  assert_eq(t[0x0A], 'Ċ')
  assert_eq(t[0x61], 'a')
  assert_eq(t[0x7F], 'ġ')
}

///|
/// Parse tokenizer.json. Fails on missing model fields.
pub fn Tokenizer::load(json : String) -> Tokenizer raise {
  let top : Json = @json.parse(json)
  guard top is Object(top) else { fail("tokenizer.json: not an object") }
  guard top["model"] is Object(m) else { fail("no model") }
  guard m["vocab"] is Object(vocab) else { fail("no vocab") }
  let id_to_tok : Array[String] = Array::make(vocab.length(), "")
  let tok_to_id : Map[String, Int] = Map([])
  for tok, idj in vocab {
    guard idj is Number(id, ..) else { fail("vocab id not number") }
    let id = id.to_int()
    id_to_tok[id] = tok
    tok_to_id[tok] = id
  }
  guard m["merges"] is Array(merges) else { fail("no merges") }
  let merge_rank : Map[String, Int] = Map([])
  for i, mj in merges {
    // both formats seen in the wild: ["a", "b"] or "a b"
    let pair = match mj {
      Array([String(a), String(b)]) => a + "\u{1}" + b
      String(s) => s
      _ => fail("bad merge entry")
    }
    merge_rank[pair] = i
  }
  // added_tokens (special tokens): decode-side lookup
  if top.get("added_tokens") is Some(Array(added)) {
    for tj in added {
      guard tj is Object(t) else { continue }
      guard t["content"] is String(content) else { continue }
      guard t["id"] is Number(id, ..) else { continue }
      let id = id.to_int()
      if id >= id_to_tok.length() {
        // sparse ids beyond base vocab
        while id_to_tok.length() <= id {
          id_to_tok.push("")
        }
      }
      id_to_tok[id] = content
      tok_to_id[content] = id
    }
  }
  let char_to_byte : Map[Char, Int] = Map([])
  let table = byte_to_unicode()
  for b in 0..<256 {
    char_to_byte[table[b]] = b
  }
  { id_to_tok, tok_to_id, merge_rank, char_to_byte }
}

// ---------- pre-tokenizer ----------
// Hand-coded equivalent of Qwen's split regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Unicode classes are approximated: ASCII is exact; c >= 0x80 counts as a
// letter unless it is known whitespace or a known punctuation block
// (general punctuation, CJK punctuation/symbols, fullwidth forms). Non-ASCII
// digits (e.g. Arabic-Indic) are treated as letters — a known divergence.

///|
fn is_ascii_letter(c : Char) -> Bool {
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

///|
fn is_digit(c : Char) -> Bool {
  c >= '0' && c <= '9'
}

///|
fn is_space(c : Char) -> Bool {
  let i = c.to_int()
  i == 0x20 ||
  (i >= 0x09 && i <= 0x0D) ||
  i == 0x85 ||
  i == 0xA0 ||
  i == 0x1680 ||
  (i >= 0x2000 && i <= 0x200A) ||
  i == 0x2028 ||
  i == 0x2029 ||
  i == 0x202F ||
  i == 0x205F ||
  i == 0x3000
}

///|
/// Non-ASCII punctuation/symbol blocks we must NOT treat as letters.
fn is_nonascii_punct(c : Char) -> Bool {
  let i = c.to_int()
  (i >= 0x2000 && i <= 0x206F) || // general punctuation
  (i >= 0x3000 && i <= 0x303F) || // CJK punctuation
  (i >= 0xFF00 && i <= 0xFF65) || // fullwidth forms
  (i >= 0x2E00 && i <= 0x2E7F) // supplemental punctuation
}

///|
fn is_letter(c : Char) -> Bool {
  if is_ascii_letter(c) {
    return true
  }
  let i = c.to_int()
  i >= 0x80 && !is_space(c) && !is_nonascii_punct(c)
}

///|
fn lower(c : Char) -> Char {
  if c >= 'A' && c <= 'Z' {
    (c.to_int() + 32).unsafe_to_char()
  } else {
    c
  }
}

///|
/// (?i:'s|'t|'re|'ve|'m|'ll|'d) at position i; returns contraction length.
fn contraction_len(cs : Array[Char], i : Int) -> Int {
  if cs[i] != '\'' {
    return 0
  }
  let n = cs.length()
  let at = fn(k) { if i + k < n { Some(lower(cs[i + k])) } else { None } }
  match at(1) {
    Some('s') | Some('t') | Some('m') | Some('d') => 2
    Some('r') => if at(2) == Some('e') { 3 } else { 0 }
    Some('v') => if at(2) == Some('e') { 3 } else { 0 }
    Some('l') => if at(2) == Some('l') { 3 } else { 0 }
    _ => 0
  }
}

///|
/// Split into pre-tokens following the Qwen regex (alternation, first match).
fn pretokenize(text : String) -> Array[String] {
  let cs = text.to_array()
  let n = cs.length()
  let out : Array[String] = []
  let mut i = 0
  while i < n {
    let c = cs[i]
    let start = i
    if contraction_len(cs, i) > 0 {
      i += contraction_len(cs, i)
    } else if is_letter(c) {
      while i < n && is_letter(cs[i]) {
        i += 1
      }
    } else if !is_digit(c) &&
      c != '\r' &&
      c != '\n' &&
      i + 1 < n &&
      is_letter(cs[i + 1]) {
      // [^\r\n\p{L}\p{N}]?\p{L}+ — punctuation/space glued to a letter run
      i += 1
      while i < n && is_letter(cs[i]) {
        i += 1
      }
    } else if is_digit(c) {
      i += 1
    } else if !is_space(c) && !is_letter(c) && !is_digit(c) {
      // [^\s\p{L}\p{N}]+[\r\n]* — punctuation run + trailing newlines
      while i < n && !is_space(cs[i]) && !is_letter(cs[i]) && !is_digit(cs[i]) {
        i += 1
      }
      while i < n && (cs[i] == '\r' || cs[i] == '\n') {
        i += 1
      }
    } else if c == ' ' &&
      i + 1 < n &&
      !is_space(cs[i + 1]) &&
      !is_letter(cs[i + 1]) &&
      !is_digit(cs[i + 1]) {
      // " ?" prefix of the punctuation rule
      i += 1
      while i < n && !is_space(cs[i]) && !is_letter(cs[i]) && !is_digit(cs[i]) {
        i += 1
      }
      while i < n && (cs[i] == '\r' || cs[i] == '\n') {
        i += 1
      }
    } else {
      // whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
      while i < n && is_space(cs[i]) {
        i += 1
      }
      let run_has_newline = {
        let mut f = false
        for k in start..<i {
          if cs[k] == '\r' || cs[k] == '\n' {
            f = true
          }
        }
        f
      }
      // \s+(?!\S) backtracks one only when the run is 2+ chars; a single
      // space before a digit/punct stays its own token (plain \s+)
      if !run_has_newline && i - start >= 2 && i < n && !is_space(cs[i]) {
        i -= 1
      }
    }
    out.push(String::from_array(cs[start:i].iter().collect()))
  }
  out
}

// ---------- BPE ----------

///|
/// Map a pre-token's chars to the byte-level token string.
fn to_byte_level(s : String) -> String {
  let bytes = @utf8.encode(s)
  let sb = StringBuilder::new()
  let table = byte_to_unicode()
  for b in bytes {
    sb.write_char(table[b.to_int()])
  }
  sb.to_string()
}

///|
/// BPE-merge one byte-level word into vocab token strings.
fn Tokenizer::bpe(self : Tokenizer, word : String) -> Array[String] {
  let mut parts : Array[String] = word
    .iter()
    .map(fn(c) { c.to_string() })
    .collect()
  if parts.length() <= 1 {
    return parts
  }
  for ;; {
    // find lowest-rank adjacent pair
    let mut best_rank = @int.MAX_VALUE
    let mut best_i = -1
    for k in 0..<(parts.length() - 1) {
      match self.merge_rank.get(parts[k] + "\u{1}" + parts[k + 1]) {
        Some(r) =>
          if r < best_rank {
            best_rank = r
            best_i = k
          }
        None => ()
      }
    }
    if best_i < 0 {
      break
    }
    let merged = parts[best_i] + parts[best_i + 1]
    let next : Array[String] = []
    let mut k = 0
    while k < parts.length() {
      if k + 1 < parts.length() && parts[k] + parts[k + 1] == merged {
        next.push(merged)
        k += 2
      } else {
        next.push(parts[k])
        k += 1
      }
    }
    parts = next
    if parts.length() <= 1 {
      break
    }
  }
  parts
}

///|
/// Encode text to token ids. Special tokens in the input are not recognized.
pub fn Tokenizer::encode(self : Tokenizer, text : String) -> Array[Int] raise {
  let out : Array[Int] = []
  for pre in pretokenize(text) {
    let word = to_byte_level(pre)
    for tok in self.bpe(word) {
      match self.tok_to_id.get(tok) {
        Some(id) => out.push(id)
        None => fail("token not in vocab: \{tok.escape()}")
      }
    }
  }
  out
}

///|
/// Decode token ids back to text (lossy UTF-8).
pub fn Tokenizer::decode(self : Tokenizer, ids : Array[Int]) -> String raise {
  let bytes : Array[Byte] = []
  for id in ids {
    bytes.append(self.token_bytes(id))
  }
  @utf8.decode_lossy(Bytes::from_array(bytes)[:])
}

///|
/// Raw bytes of one token (special tokens map to nothing).
pub fn Tokenizer::token_bytes(self : Tokenizer, id : Int) -> Array[Byte] raise {
  guard id >= 0 && id < self.id_to_tok.length() else { fail("bad id \{id}") }
  let tok = self.id_to_tok[id]
  let out : Array[Byte] = []
  if tok.has_prefix("<|") && tok.has_suffix("|>") {
    return out // skip special tokens
  }
  for c in tok {
    match self.char_to_byte.get(c) {
      Some(b) => out.push(b.to_byte())
      None => ()
    }
  }
  out
}

///|
/// Incremental UTF-8-safe decoder for streaming generation: multi-byte
/// characters split across tokens are held back until complete. A literal
/// U+FFFD stored in the vocab is indistinguishable from an incomplete tail
/// and would be held back forever (accepted edge case).
pub struct StreamDecoder {
  tk : Tokenizer
  buf : Array[Byte]
  mut shown : Int // chars already emitted
}

///|
pub fn StreamDecoder::new(tk : Tokenizer) -> StreamDecoder {
  { tk, buf: [], shown: 0 }
}

///|
/// Append one token; returns the newly-completed text ("" while a UTF-8
/// sequence is still incomplete).
pub fn StreamDecoder::push(self : StreamDecoder, id : Int) -> String raise {
  self.buf.append(self.tk.token_bytes(id))
  let cs = @utf8.decode_lossy(Bytes::from_array(self.buf)[:]).to_array()
  let mut safe = cs.length()
  while safe > self.shown && cs[safe - 1] == '\u{FFFD}' {
    safe -= 1
  }
  if safe == self.shown {
    return ""
  }
  let out = String::from_array(cs[self.shown:safe].iter().collect())
  self.shown = safe
  out
}

///|
test "decode round-trips byte-level tokens" {
  // minimal hand-built tokenizer
  let json =
    #|{"model":{"vocab":{"Ġhello":1,"Ċ":2,"a":0},"merges":[]}}
  let t = Tokenizer::load(json)
  assert_eq(t.decode([1, 0, 2]), " helloa\n")
}

///|
test "StreamDecoder holds incomplete UTF-8" {
  let t = byte_to_unicode()
  // token a = bytes [E4 B8] (first two of 中), token b = [AD]
  let ta = String::from_array([t[0xE4], t[0xB8]])
  let tb = String::from_array([t[0xAD]])
  let json =
    $|{"model":{"vocab":{"\{ta}":0,"\{tb}":1,"a":2,"<|x|>":3},"merges":[]},
    $| "added_tokens":[{"id":3,"content":"<|x|>"}]}
  let tk = Tokenizer::load(json)
  let sd = StreamDecoder::new(tk)
  assert_eq(sd.push(0), "") // E4 B8 alone is incomplete
  assert_eq(sd.push(1), "中")
  assert_eq(sd.push(2), "a")
  assert_eq(sd.push(3), "") // special token: no output
}

///|
test "StreamDecoder plain ascii streams" {
  let json =
    #|{"model":{"vocab":{"he":0,"llo":1},"merges":[]}}
  let tk = Tokenizer::load(json)
  let sd = StreamDecoder::new(tk)
  assert_eq(sd.push(0), "he")
  assert_eq(sd.push(1), "llo")
}
