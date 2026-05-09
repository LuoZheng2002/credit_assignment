use credit_assignment::agent::state_to_actions::detect_repetition_five_times;

#[test]
fn detects_exact_three_consecutive_repetitions() {
    let unit = "12345678901234567890";
    let response = format!("prefix {}{}{} suffix", unit, unit, unit);
    assert!(detect_repetition_five_times(&response));
}

#[test]
fn detects_more_than_three_consecutive_repetitions() {
    let unit = "ABCDEFGHIJKLMNOPQRST";
    let response = format!("{}{}{}{}", unit, unit, unit, unit);
    assert!(detect_repetition_five_times(&response));
}

#[test]
fn does_not_trigger_when_only_two_repetitions() {
    let unit = "abcdefghijklmnopqrst";
    let response = format!("{}{}", unit, unit);
    assert!(!detect_repetition_five_times(&response));
}

#[test]
fn does_not_trigger_for_non_consecutive_repetitions() {
    let unit = "zyxwvutsrqponmlkjihg";
    let response = format!("{} middle {} middle {}", unit, unit, unit);
    assert!(!detect_repetition_five_times(&response));
}

#[test]
fn does_not_trigger_for_short_repeating_unit_below_min_length() {
    let short_unit = "repeat_short_19_len"; // 19 bytes
    assert_eq!(short_unit.len(), 19);
    let response = format!("{}{}{}", short_unit, short_unit, short_unit);
    assert!(!detect_repetition_five_times(&response));
}

#[test]
fn does_not_trigger_for_overlapping_pattern_only() {
    let response = "a".repeat(59);
    assert!(!detect_repetition_five_times(&response));
}
