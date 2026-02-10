//! Diff computation for detecting changed blocks between file versions.
//!
//! This module provides a port of Python's difflib.SequenceMatcher for
//! computing line-based diffs between two strings.

use std::collections::HashMap;

/// Represents a changed block with line numbers (1-based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedBlock {
    /// Start line in 'before' (1-based)
    pub start_before: usize,
    /// End line in 'before' (1-based, inclusive)
    pub end_before: usize,
    /// Start line in 'after' (1-based)
    pub start_after: usize,
    /// End line in 'after' (1-based, inclusive)
    pub end_after: usize,
    /// The replacement lines from 'after'
    pub replacement_lines: Vec<String>,
}

/// Opcode tag indicating the type of operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpcodeTag {
    Replace,
    Delete,
    Insert,
    Equal,
}

/// An opcode representing a single edit operation.
/// Format: (tag, i1, i2, j1, j2) where:
/// - i1:i2 is the range in sequence a
/// - j1:j2 is the range in sequence b
type Opcode = (OpcodeTag, usize, usize, usize, usize);

/// A matching block: (i, j, n) means a[i:i+n] == b[j:j+n]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Match {
    i: usize,
    j: usize,
    n: usize,
}

/// Port of Python's difflib.SequenceMatcher with autojunk=False.
struct SequenceMatcher<'a> {
    a: Vec<&'a str>,
    b: Vec<&'a str>,
    b2j: HashMap<&'a str, Vec<usize>>,
}

impl<'a> SequenceMatcher<'a> {
    fn new(a: Vec<&'a str>, b: Vec<&'a str>) -> Self {
        let mut b2j: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, &elt) in b.iter().enumerate() {
            b2j.entry(elt).or_default().push(i);
        }
        Self { a, b, b2j }
    }

    /// Find longest matching block in a[alo:ahi] and b[blo:bhi].
    fn find_longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> Match {
        let mut besti = alo;
        let mut bestj = blo;
        let mut bestsize = 0;

        let mut j2len: HashMap<usize, usize> = HashMap::new();

        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(indices) = self.b2j.get(self.a[i]) {
                for &j in indices {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j2len.get(&(j.wrapping_sub(1))).unwrap_or(&0) + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }

        // Extend match backwards
        while besti > alo && bestj > blo && self.a[besti - 1] == self.b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }

        // Extend match forwards
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && self.a[besti + bestsize] == self.b[bestj + bestsize]
        {
            bestsize += 1;
        }

        Match {
            i: besti,
            j: bestj,
            n: bestsize,
        }
    }

    /// Return list of matching blocks.
    fn get_matching_blocks(&self) -> Vec<Match> {
        let la = self.a.len();
        let lb = self.b.len();

        let mut queue = vec![(0, la, 0, lb)];
        let mut matching_blocks = Vec::new();

        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let m = self.find_longest_match(alo, ahi, blo, bhi);
            if m.n > 0 {
                matching_blocks.push(m);
                if alo < m.i && blo < m.j {
                    queue.push((alo, m.i, blo, m.j));
                }
                if m.i + m.n < ahi && m.j + m.n < bhi {
                    queue.push((m.i + m.n, ahi, m.j + m.n, bhi));
                }
            }
        }

        // Sort by (i, j, n)
        matching_blocks.sort_by(|a, b| {
            a.i.cmp(&b.i)
                .then_with(|| a.j.cmp(&b.j))
                .then_with(|| a.n.cmp(&b.n))
        });

        // Collapse adjacent equal blocks
        let mut i1 = 0;
        let mut j1 = 0;
        let mut k1 = 0;
        let mut result = Vec::new();

        for m in matching_blocks {
            if i1 + k1 == m.i && j1 + k1 == m.j {
                k1 += m.n;
            } else {
                if k1 > 0 {
                    result.push(Match {
                        i: i1,
                        j: j1,
                        n: k1,
                    });
                }
                i1 = m.i;
                j1 = m.j;
                k1 = m.n;
            }
        }
        if k1 > 0 {
            result.push(Match {
                i: i1,
                j: j1,
                n: k1,
            });
        }

        // Append sentinel
        result.push(Match {
            i: la,
            j: lb,
            n: 0,
        });

        result
    }

    /// Return list of opcodes describing how to turn a into b.
    fn get_opcodes(&self) -> Vec<Opcode> {
        let mut opcodes = Vec::new();
        let mut i = 0;
        let mut j = 0;

        for m in self.get_matching_blocks() {
            let mut tag = None;

            if i < m.i && j < m.j {
                tag = Some(OpcodeTag::Replace);
            } else if i < m.i {
                tag = Some(OpcodeTag::Delete);
            } else if j < m.j {
                tag = Some(OpcodeTag::Insert);
            }

            if let Some(t) = tag {
                opcodes.push((t, i, m.i, j, m.j));
            }
            if m.n > 0 {
                opcodes.push((OpcodeTag::Equal, m.i, m.i + m.n, m.j, m.j + m.n));
            }
            i = m.i + m.n;
            j = m.j + m.n;
        }

        opcodes
    }
}

/// Compute a unified diff between two strings with context lines.
///
/// Returns a string in unified diff format (without the file headers).
pub fn compute_unified_diff(old_content: &str, new_content: &str, context_lines: usize) -> String {
    let old_lines: Vec<&str> = old_content.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    if old_lines == new_lines {
        return String::new();
    }

    let sm = SequenceMatcher::new(old_lines.clone(), new_lines.clone());
    let opcodes = sm.get_opcodes();

    // Group opcodes into hunks with context
    let mut hunks: Vec<String> = Vec::new();
    let mut i = 0;

    while i < opcodes.len() {
        // Skip leading equal blocks
        if opcodes[i].0 == OpcodeTag::Equal {
            i += 1;
            continue;
        }

        // Find hunk boundaries
        let mut hunk_start_i = i;
        let mut hunk_end_i = i;

        // Look for a contiguous group of changes (with context overlap)
        while hunk_end_i < opcodes.len() {
            if opcodes[hunk_end_i].0 == OpcodeTag::Equal {
                // Check if next non-equal is within context range
                let equal_len = opcodes[hunk_end_i].2 - opcodes[hunk_end_i].1;
                if equal_len > context_lines * 2 {
                    break;
                }
            }
            hunk_end_i += 1;
        }

        // Build hunk
        let first_op = &opcodes[hunk_start_i];
        let last_op = &opcodes[hunk_end_i.saturating_sub(1).max(hunk_start_i)];

        // Calculate context bounds
        let old_start = first_op.1.saturating_sub(context_lines);
        let old_end = (last_op.2 + context_lines).min(old_lines.len());
        let new_start = first_op.3.saturating_sub(context_lines);
        let new_end = (last_op.4 + context_lines).min(new_lines.len());

        let old_len = old_end - old_start;
        let new_len = new_end - new_start;

        let mut hunk_lines: Vec<String> = Vec::new();
        hunk_lines.push(format!(
            "@@ -{},{} +{},{} @@",
            old_start + 1,
            old_len,
            new_start + 1,
            new_len
        ));

        // Add context before
        for line in old_lines.iter().take(first_op.1).skip(old_start) {
            hunk_lines.push(format!(" {}", line));
        }

        // Add changes
        for op_idx in hunk_start_i..hunk_end_i {
            let op = &opcodes[op_idx];
            match op.0 {
                OpcodeTag::Replace => {
                    for line in old_lines.iter().take(op.2).skip(op.1) {
                        hunk_lines.push(format!("-{}", line));
                    }
                    for line in new_lines.iter().take(op.4).skip(op.3) {
                        hunk_lines.push(format!("+{}", line));
                    }
                }
                OpcodeTag::Delete => {
                    for line in old_lines.iter().take(op.2).skip(op.1) {
                        hunk_lines.push(format!("-{}", line));
                    }
                }
                OpcodeTag::Insert => {
                    for line in new_lines.iter().take(op.4).skip(op.3) {
                        hunk_lines.push(format!("+{}", line));
                    }
                }
                OpcodeTag::Equal => {
                    for line in old_lines.iter().take(op.2).skip(op.1) {
                        hunk_lines.push(format!(" {}", line));
                    }
                }
            }
        }

        // Add context after
        for line in old_lines.iter().take(old_end).skip(last_op.2) {
            hunk_lines.push(format!(" {}", line));
        }

        hunks.push(hunk_lines.join("\n"));
        i = hunk_end_i;
    }

    hunks.join("\n")
}

/// Compute the changed block between two strings.
///
/// Returns 1-based line numbers for the changed region and the replacement lines.
pub fn compute_changed_block_lines(before: &str, after: &str) -> Result<ChangedBlock, &'static str> {
    let before_lines: Vec<&str> = before.lines().collect();
    let after_lines: Vec<&str> = after.lines().collect();

    let sm = SequenceMatcher::new(before_lines.clone(), after_lines.clone());
    let all_opcodes = sm.get_opcodes();
    let non_equal: Vec<_> = all_opcodes
        .into_iter()
        .filter(|(tag, _, _, _, _)| *tag != OpcodeTag::Equal)
        .collect();

    if non_equal.is_empty() {
        return Err("Opcode list cannot be empty! Likely a bug in the diff computation.");
    }

    let first = non_equal.first().unwrap();
    let last = non_equal.last().unwrap();

    // i1/i2 refer to 'before' indices, j1/j2 to 'after'
    let start_before = (first.1 + 1).max(1);
    let end_before = last.2;
    let start_after = (first.3 + 1).max(1);
    let end_after = last.4;
    let replacement_lines: Vec<String> = after_lines[first.3..last.4]
        .iter()
        .map(|s| s.to_string())
        .collect();

    Ok(ChangedBlock {
        start_before,
        end_before,
        start_after,
        end_after,
        replacement_lines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_changed_block_simple_replace() {
        let before = "line1\nline2\nline3";
        let after = "line1\nmodified\nline3";
        let result = compute_changed_block_lines(before, after).unwrap();
        assert_eq!(result.start_before, 2);
        assert_eq!(result.end_before, 2);
        assert_eq!(result.replacement_lines, vec!["modified"]);
    }

    #[test]
    fn test_compute_changed_block_insert() {
        let before = "line1\nline3";
        let after = "line1\nline2\nline3";
        let result = compute_changed_block_lines(before, after).unwrap();
        assert!(result.replacement_lines.contains(&"line2".to_string()));
    }

    #[test]
    fn test_compute_changed_block_delete() {
        let before = "line1\nline2\nline3";
        let after = "line1\nline3";
        let result = compute_changed_block_lines(before, after).unwrap();
        assert_eq!(result.start_before, 2);
        assert_eq!(result.end_before, 2);
    }
}

