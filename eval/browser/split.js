/*
 * Held-out split helpers for the Mixtral Healthcare evaluation.
 * Re-implements, in JavaScript, the deterministic splits used at training time:
 *   mixtral_lab/healthcare_corpus.py  row_id() + stable_partition()   (pretraining, 2% validation)
 *   mixtral_lab/medical_data.py       bucket_for_question()            (SFT, 5% validation)
 * BLAKE2b is a compact 32-bit-limb implementation (RFC 7693), checked against Python's hashlib.
 */
var Split = (function () {
  'use strict';
  var IV = new Uint32Array([
    0xF3BCC908, 0x6A09E667, 0x84CAA73B, 0xBB67AE85, 0xFE94F82B, 0x3C6EF372, 0x5F1D36F1, 0xA54FF53A,
    0xADE682D1, 0x510E527F, 0x2B3E6C1F, 0x9B05688C, 0xFB41BD6B, 0x1F83D9AB, 0x137E2179, 0x5BE0CD19]);
  var SIGMA = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
    11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
    7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
    9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
    2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
    12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
    13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
    6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
    10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3].map(function (x) { return x * 2; });
  var v = new Uint32Array(32), m = new Uint32Array(32);

  function add64aa(a, b) {
    var o0 = v[a] + v[b], o1 = v[a + 1] + v[b + 1];
    if (o0 >= 0x100000000) o1++;
    v[a] = o0; v[a + 1] = o1;
  }
  function add64ac(a, b0, b1) {
    var o0 = v[a] + b0, o1 = v[a + 1] + b1;
    if (o0 >= 0x100000000) o1++;
    v[a] = o0; v[a + 1] = o1;
  }
  function G(a, b, c, d, ix, iy) {
    var x0 = m[ix], x1 = m[ix + 1], y0 = m[iy], y1 = m[iy + 1], t0, t1;
    add64aa(a, b); add64ac(a, x0, x1);
    t0 = v[d] ^ v[a]; t1 = v[d + 1] ^ v[a + 1]; v[d] = t1; v[d + 1] = t0;                 // rotr 32
    add64aa(c, d);
    t0 = v[b] ^ v[c]; t1 = v[b + 1] ^ v[c + 1];
    v[b] = (t0 >>> 24) ^ (t1 << 8); v[b + 1] = (t1 >>> 24) ^ (t0 << 8);                      // rotr 24
    add64aa(a, b); add64ac(a, y0, y1);
    t0 = v[d] ^ v[a]; t1 = v[d + 1] ^ v[a + 1];
    v[d] = (t0 >>> 16) ^ (t1 << 16); v[d + 1] = (t1 >>> 16) ^ (t0 << 16);                    // rotr 16
    add64aa(c, d);
    t0 = v[b] ^ v[c]; t1 = v[b + 1] ^ v[c + 1];
    v[b] = (t1 >>> 31) ^ (t0 << 1); v[b + 1] = (t0 >>> 31) ^ (t1 << 1);                      // rotr 63
  }
  function compress(ctx, last) {
    var i;
    for (i = 0; i < 16; i++) { v[i] = ctx.h[i]; v[i + 16] = IV[i]; }
    v[24] ^= ctx.t; v[25] ^= (ctx.t / 0x100000000);
    if (last) { v[28] = ~v[28]; v[29] = ~v[29]; }
    for (i = 0; i < 32; i++) {
      var p = 4 * i;
      m[i] = ctx.b[p] ^ (ctx.b[p + 1] << 8) ^ (ctx.b[p + 2] << 16) ^ (ctx.b[p + 3] << 24);
    }
    for (i = 0; i < 12; i++) {
      var s = i * 16;
      G(0, 8, 16, 24, SIGMA[s], SIGMA[s + 1]);
      G(2, 10, 18, 26, SIGMA[s + 2], SIGMA[s + 3]);
      G(4, 12, 20, 28, SIGMA[s + 4], SIGMA[s + 5]);
      G(6, 14, 22, 30, SIGMA[s + 6], SIGMA[s + 7]);
      G(0, 10, 20, 30, SIGMA[s + 8], SIGMA[s + 9]);
      G(2, 12, 22, 24, SIGMA[s + 10], SIGMA[s + 11]);
      G(4, 14, 16, 26, SIGMA[s + 12], SIGMA[s + 13]);
      G(6, 8, 18, 28, SIGMA[s + 14], SIGMA[s + 15]);
    }
    for (i = 0; i < 16; i++) ctx.h[i] = ctx.h[i] ^ v[i] ^ v[i + 16];
  }
  function blake2b(bytes, outlen) {
    var ctx = { h: new Uint32Array(IV), b: new Uint8Array(128), c: 0, t: 0 };
    ctx.h[0] ^= 0x01010000 ^ outlen;
    for (var i = 0; i < bytes.length; i++) {
      if (ctx.c === 128) { ctx.t += ctx.c; compress(ctx, false); ctx.c = 0; }
      ctx.b[ctx.c++] = bytes[i];
    }
    ctx.t += ctx.c;
    while (ctx.c < 128) ctx.b[ctx.c++] = 0;
    compress(ctx, true);
    var out = new Uint8Array(outlen);
    for (i = 0; i < outlen; i++) out[i] = ctx.h[i >> 2] >> (8 * (i & 3));
    return out;
  }
  var enc = new TextEncoder();
  function hex(u8) { var s = ''; for (var i = 0; i < u8.length; i++) s += (u8[i] < 16 ? '0' : '') + u8[i].toString(16); return s; }
  // Python: int.from_bytes(digest, "big") % 100
  function mod100(u8) { var r = 0; for (var i = 0; i < u8.length; i++) r = (r * 256 + u8[i]) % 100; return r; }

  // Python str.split() whitespace set, and the str.casefold() cases that differ from toLowerCase().
  var WS = new RegExp('[\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]+');
  var FOLD = { 'ß': 'ss', 'ẞ': 'ss', 'ς': 'σ', 'ſ': 's', 'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'st', 'ﬆ': 'st', 'µ': 'μ' };
  function casefold(s) { return s.toLowerCase().replace(/[ßẞςſﬀ-ﬆµ]/g, function (c) { return FOLD[c] || c; }); }
  function canonical(s) { return casefold(s).split(WS).filter(Boolean).join(' '); }

  return {
    blake2bHex: function (str, outlen) { return hex(blake2b(enc.encode(str), outlen)); },
    canonical: canonical,
    // healthcare_corpus.stable_partition(row_id(text)), seed 42, 2% validation
    pretrainPartition: function (text, seed, pct) {
      var docHash = hex(blake2b(enc.encode(canonical(text)), 16));
      return mod100(blake2b(enc.encode((seed == null ? 42 : seed) + ':' + docHash), 8)) < (pct == null ? 2 : pct) ? 'validation' : 'train';
    },
    // medical_data.bucket_for_question(question), seed 42, 5% validation
    sftBucket: function (question, seed, pct) {
      return mod100(blake2b(enc.encode((seed == null ? 42 : seed) + ':' + canonical(question)), 8)) < (pct == null ? 5 : pct) ? 'validation' : 'train';
    }
  };
})();
if (typeof module !== 'undefined') module.exports = Split;
