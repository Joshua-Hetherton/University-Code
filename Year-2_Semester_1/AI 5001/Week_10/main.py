def solve_consecutive_puzzle():
    """
    Solve: Sum of any 3 consecutive numbers = 11
    Given numbers: 2 and 5
    """
    
    # The key insight: if a+b+c=11 and b+c+d=11, then d=a
    # So the sequence repeats every 3 numbers: [x, y, z, x, y, z, ...]
    
    # Let the pattern be [a, b, c] where a + b + c = 11
    # We need to find a, b, c such that the pattern contains 2 and 5
    
    def find_pattern_with_numbers():
        """Find the repeating pattern [a, b, c] that contains 2 and 5"""
        solutions = []
        
        # Try all combinations where 2 and 5 are in the pattern
        for pos_2 in [0, 1, 2]:
            for pos_5 in [0, 1, 2]:
                if pos_2 != pos_5:
                    pattern = [0, 0, 0]
                    pattern[pos_2] = 2
                    pattern[pos_5] = 5
                    
                    # Find the third number
                    used_positions = [pos_2, pos_5]
                    third_pos = [p for p in [0, 1, 2] if p not in used_positions][0]
                    pattern[third_pos] = 11 - 2 - 5  # 11 - 7 = 4
                    
                    # Verify the sum
                    if sum(pattern) == 11:
                        solutions.append(pattern)
        
        return solutions
    
    # Find all possible patterns
    patterns = find_pattern_with_numbers()
    
    if patterns:
        print("Found possible repeating patterns:")
        for i, pattern in enumerate(patterns, 1):
            print(f"Pattern {i}: {pattern}")
        
        # Let's use the first pattern and create a complete sequence
        chosen_pattern = patterns[0]
        print(f"\nUsing pattern: {chosen_pattern}")
        
        # Create a longer sequence to show the complete puzzle
        sequence_length = 8  # Common length for such puzzles
        full_sequence = []
        
        for i in range(sequence_length):
            full_sequence.append(chosen_pattern[i % 3])
        
        print(f"\nComplete sequence (length {sequence_length}):")
        print(f"{full_sequence}")
        
        # Verify the solution
        print(f"\nVerification (sum of every 3 consecutive numbers):")
        for i in range(len(full_sequence) - 2):
            triple = full_sequence[i:i+3]
            print(f"Positions {i+1}-{i+3}: {triple} = sum({sum(triple)})")
        
        return full_sequence, chosen_pattern
    
    return None, None

# Solve the puzzle
print("PUZZLE: Fill in to make the sum of any 3 consecutive numbers = 11")
print("Given numbers: 2 and 5\n")

sequence, pattern = solve_consecutive_puzzle()

if sequence:
    print(f"\n=== FINAL SOLUTION ===")
    print(f"Repeating pattern: {pattern}")
    print(f"Complete sequence: {sequence}")
    
    # Display in a more visual format
    print(f"\nVisual representation:")
    for i, num in enumerate(sequence, 1):
        print(f"Position {i}: {num}")
    
    print(f"\nPattern explanation:")
    print(f"The sequence repeats every 3 numbers: {pattern}")
    print(f"This guarantees that ANY 3 consecutive numbers sum to 11")
else:
    print("No solution found!")