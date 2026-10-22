"""CPU-only lexical adapter contract. Never compiles or runs HIP.

Run from any directory. --candidate FILE checks the three function bodies named
below in a future private translation unit. A PASS is source correspondence,
not C++ parsing, binding resolution, native arithmetic or GPU qualification.
No candidate mathematical kernel is written by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'reports/grouped-k3-adapter-pins.json'
BASELINE = 'a5fbb60a5cde97442455e041e7f10c76397209cf2c335fc16600346c3da59503'
SOURCES = {
    'input': ('src/transforms.hpp', 'input_transform', 'grouped_input_transform'),
    'core': ('src/gemv_lds.hip', 'gemv_baseline_kernel', 'grouped_baseline_core'),
    'output': ('src/transforms.hpp', 'output_transform', 'grouped_output_transform'),
}
# Deliberately exact, finite binding list. Do not replace arbitrary identifiers
# or simplify expressions in the retained old body.
PROLOGUES = {
    'input': '''const int batch_group=blockIdx.y;
const auto& g=b.groups[batch_group];
const float* input=g.x; const uint16_t* scales=g.suh;
_Float16* transformed=g.a; const int input_size=b.input,rows=1;''',
    'core': '''const int batch_group=blockIdx.z;
const auto& g=b.groups[batch_group];
const uint32_t* __restrict__ packed=g.packed; const _Float16* __restrict__ x=g.a;
float* __restrict__ partial=g.partial; const int input=b.input,output=b.output;
const int rows=1,total_rows=1,splits=b.splits,codebook=2;
constexpr int Bits=3,Rows=1; constexpr bool Mul1=true,Fast=false;
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));''',
    'output': '''const int batch_group=blockIdx.y;
const auto& g=b.groups[batch_group];
const float* partial=g.partial; const uint16_t* scales=g.svh;
const float* bias=g.bias; float* output=g.y;
const int output_size=b.output,rows=1,splits=b.splits;
const float alpha=g.alpha,beta=g.beta;''',
}
# A restricted C++ lexer, including literals and multi-character operators.
# It retains preprocessor tokens but does not implement preprocessing.
LEX = re.compile(r'''(?P<skip>\s+|//[^\n]*|/\*[\s\S]*?\*/)|
    (?P<literal>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|
    (?P<word>[A-Za-z_]\w*|(?:0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?(?:[pP][+-]?\d+)?|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)[A-Za-z0-9_]*)|
    (?P<op>::|->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|&=|\|=|\^=|[{}()[\];,.?:~!+*/%&|^<>=#-])''', re.X)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tokens(text):
    result, pos = [], 0
    while pos < len(text):
        match = LEX.match(text, pos)
        if not match:
            raise AssertionError(f'unsupported lexical input at {text[pos:pos+32]!r}')
        if match.lastgroup != 'skip':
            result.append(match[0])
        pos = match.end()
    return result


def body(text, symbol):
    ts = tokens(text)
    found = []
    for i, token in enumerate(ts):
        if token != symbol or ts[i+1:i+2] != ['(']:
            continue
        j, depth = i+1, 0
        while j < len(ts):
            depth += (ts[j] == '(') - (ts[j] == ')')
            j += 1
            if depth == 0:
                break
        if ts[j:j+1] != ['{']:
            continue
        begin, depth = j+1, 1
        j += 1
        while j < len(ts) and depth:
            depth += (ts[j] == '{') - (ts[j] == '}')
            j += 1
        assert depth == 0, f'unbalanced body: {symbol}'
        found.append(ts[begin:j-1])
    assert len(found) == 1, f'expected one definition of {symbol}, got {len(found)}'
    return found[0]


def same_body(stage, actual, original):
    prefix = tokens(PROLOGUES[stage])
    assert actual[:len(prefix)] == prefix, f'{stage}: binding prologue changed'
    retained = actual[len(prefix):]
    assert retained == original, f'{stage}: retained math tokens changed'


def expect_rejection(stage, mutated, original, label):
    try:
        same_body(stage, tokens(mutated), original)
    except AssertionError:
        print('REJECT mutation=' + label)
        return
    raise AssertionError('mutation was accepted: ' + label)


def check_mappings():
    # Evaluate old row-local address formulas, including inactive H128 waves,
    # independently of token comparison. No floating-point/GPU emulation.
    total = 0
    for i, n, s in ((5120, 2304, 16), (2304, 5120, 4)):
        ends = [(i//16)*k//s for k in range(s+1)]
        assert ends[0] == 0 and ends[-1] == i//16
        assert all(ends[k+1]-ends[k] == (20 if s == 16 else 36) for k in range(s))
        for count in (1, 2, 4, 8, 16, 31, 32):
            for width in (i, n):
                active = [4*x+w for x in range((width+511)//512) for w in range(4)
                          if 4*x+w < width//128]
                assert active == list(range(width//128))
                inactive = 4*((width+511)//512)-len(active)
                assert inactive == (2 if width == 2304 else 0)
            for group in range(count):
                # group selects the allocation; it never appears in its offset.
                offsets = [split*n + tile*16 + lane//4 + h*8
                           for split in range(s) for tile in range(n//16)
                           for lane in range(0, 32, 4) for h in range(2)]
                assert len(offsets) == s*n and len(set(offsets)) == s*n
                assert min(offsets) == 0 and max(offsets) == s*n-1
                total += len(offsets)
            if count > 1:
                assert ((s-1)*count+count-1)*n+n-1 >= s*n  # wrong total_rows=G
        print(f'MAPPING shape={i},{n},{s} G=1,2,4,8,16,31,32 endpoints/tails/localP=PASS')
    print(f'MAPPING checked_P_writers={total}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--helper-dir', type=Path,
                        help='Optional directory of copied own helper headers; exact tokens required')
    args = parser.parse_args()
    pins = json.loads(PINS.read_text())
    for name, expected in pins['files_lf'].items():
        assert sha((ROOT/name).read_text(encoding='utf-8').encode()) == expected, name
    text = (ROOT/'src/gemv_lds.hip').read_text()
    literal = text[text.index('template<int Bits, bool Mul1, int Rows, bool Fast=false>'):
                   text.index('template<int Bits,bool Mul1,bool Fast=false>')]
    assert sha(literal.encode()) == BASELINE
    print('BASELINE_LF_SHA256=' + BASELINE)
    originals = {stage: body((ROOT/path).read_text(), name)
                 for stage, (path, name, _) in SOURCES.items()}
    for stage, old in originals.items():
        assert sha(' '.join(old).encode()) == pins['body_tokens'][stage]
        adapted = PROLOGUES[stage] + '\n' + ' '.join(old)
        same_body(stage, tokens(adapted), old)
        same_body(stage, body('void cpu_fragment() {' + adapted + '}', 'cpu_fragment'), old)
        # Whitespace/comments are allowed, arithmetic/operator changes are not.
        same_body(stage, tokens('/* CPU selftest */\n' + adapted), old)
        print(f'TOKENS stage={stage} retained={len(old)} exact=PASS')
    mutations = [
        ('input', 'materialize_fp32 (', '( ', 'remove_RN32_boundary'),
        ('input', 'float ( _Float16 ( scaled ) )', 'scaled', 'remove_RN16_input'),
        ('input', 'hadamard128', 'hadamard64', 'H64'),
        ('input', 'blockIdx.y', 'blockIdx.x', 'wrong_input_group'),
        ('core', 'total_rows=1', 'total_rows=b.count', 'global_row_stride'),
        ('core', 'Fast=false', 'Fast=true', 'experimental_reader'),
        ('core', 'blockIdx.z', 'blockIdx.y', 'wrong_core_group'),
        ('core', '( blockIdx . y + 1 ) / splits', 'blockIdx . y / splits + 1', 'split_end'),
        ('core', 'sums [ Rows ] [ 2 ]', 'sums [ Rows ] [ 4 ]', 'sum_shape'),
        ('core', 'false ) ;', 'true ) ;', 'fdot2_clamp'),
        ('core', '__shfl_down ( v , 2 , 4 )', '__shfl_down ( v , 1 , 4 )', 'reduction_order'),
        ('output', 'beta != 0', 'beta > 0', 'negative_beta_skipped'),
        ('output', 'if ( beta != 0 )', 'if ( true )', 'zero_beta_reads_NaN'),
        ('output', 'value *= alpha', 'value += alpha', 'epilogue_math'),
        ('output', '++ split', 'split += 2', 'missing_split'),
    ]
    for stage, before, after, label in mutations:
        source = PROLOGUES[stage] + '\n' + ' '.join(originals[stage])
        assert before in source, label
        expect_rejection(stage, source.replace(before, after, 1), originals[stage], label)
    # Helper tokens also retain constants and inline asm strings, not just calls.
    helpers = ('hadamard.hpp', 'trellis_optimized.hpp', 'trellis_device.hpp', 'packed_decoder.hpp')
    for name in helpers:
        expected = tokens((ROOT/'src'/name).read_text())
        assert sha(' '.join(expected).encode()) == pins['helper_tokens'][name]
        if args.helper_dir:
            assert tokens((args.helper_dir/name).read_text()) == expected, f'helper changed: {name}'
    print(f'HELPERS canonical_files={len(helpers)} pinned_tokens=PASS copied_headers=' +
          ('PASS' if args.helper_dir else 'NOT_REQUESTED'))
    helper = tokens((ROOT/'src/hadamard.hpp').read_text())
    altered = tokens((ROOT/'src/hadamard.hpp').read_text().replace('0x1.6a09e6p-4f', '0x1p-4f'))
    try:
        assert altered == helper
    except AssertionError:
        pass
    else:
        raise AssertionError('helper mutation was accepted')
    print('REJECT mutation=helper_H_normalization')
    check_mappings()
    if args.candidate:
        candidate = args.candidate.read_text(encoding='utf-8')
        for stage, (_, _, symbol) in SOURCES.items():
            same_body(stage, body(candidate, symbol), originals[stage])
        print('CANDIDATE bodies/prologues=PASS; signatures/includes/bindings/ELF/native remain OPEN')
    else:
        print('CANDIDATE=ABSENT; selftests do not close mathematical extraction')
    print('PASS CPU lexical/mapping contract only; no native math/resources/performance claim')


if __name__ == '__main__':
    main()
