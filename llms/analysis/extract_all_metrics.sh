#!/bin/bash
#BSUB -J extract_perplexity_metrics                              # Job name
#BSUB -o logs/extract_perplexity_metrics.%J.log                  # Standard output (%J = job ID)
#BSUB -e logs/extract_perplexity_metrics.%J.log                  # Standard error
#BSUB -q h100                                           # Queue to submit to (NVIDIA H100 queue)
#BSUB -gpu "num=1:gmodel=NVIDIAH100PCIe"                # Request 1 H100 GPU
#BSUB -n 4                                              # Number of CPU cores
#BSUB -W 23:59                                          # Walltime (23 hours 59 minutes max)
#BSUB -u mm9628a@american.edu                                     # Replace with your email
#BSUB -B                                                # Send email at the beginning of the job
#BSUB -N                                                # Send email at the end of the job

# extract_all_metrics.sh - Extract metrics from all checkpoints
# Usage: ./extract_all_metrics.sh

# Activate virtual environment
source ~/myenv/bin/activate

# Configuration
BASE_DIR="./out"
OUTPUT_CSV="checkpoint_metrics_summary.csv"
PYTHON_SCRIPT="extract_checkpoint_metrics.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Checkpoint Metrics Extraction"
echo "=========================================="
echo ""
echo "Base directory: $BASE_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo ""

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}✗ Base directory not found: $BASE_DIR${NC}"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}✗ Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Remove old output CSV if exists
if [ -f "$OUTPUT_CSV" ]; then
    echo "Removing old output file..."
    rm "$OUTPUT_CSV"
fi

# Initialize counters
total_found=0
total_processed=0
total_failed=0
total_skipped=0

# Find all .pt files in subdirectories
echo "Scanning for checkpoint files..."
echo ""

# Use find to get all .pt files
while IFS= read -r -d '' checkpoint_file; do
    ((total_found++))
    
    echo "----------------------------------------"
    echo "Found checkpoint #$total_found:"
    echo "  $checkpoint_file"
    
    # Check if file is readable
    if [ ! -r "$checkpoint_file" ]; then
        echo -e "${RED}✗ Cannot read file, skipping...${NC}"
        ((total_skipped++))
        continue
    fi
    
    # Get file size to check if it's suspiciously small
    file_size=$(stat -f%z "$checkpoint_file" 2>/dev/null || stat -c%s "$checkpoint_file" 2>/dev/null)
    if [ "$file_size" -lt 1000 ]; then
        echo -e "${YELLOW}⚠ File too small ($file_size bytes), likely corrupted, skipping...${NC}"
        ((total_skipped++))
        continue
    fi
    
    echo "  File size: $(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$file_size bytes")"
    echo "  Processing..."
    
    # Run the Python script
    if python "$PYTHON_SCRIPT" \
        --checkpoint_path "$checkpoint_file" \
        --output_csv "$OUTPUT_CSV"; then
        
        echo -e "${GREEN}✓ Success${NC}"
        ((total_processed++))
    else
        echo -e "${RED}✗ Failed to process checkpoint${NC}"
        ((total_failed++))
    fi
    
    echo ""
    
done < <(find "$BASE_DIR" -type f -name "*.pt" -print0)

# Print summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total checkpoints found:     $total_found"
echo -e "${GREEN}Successfully processed:      $total_processed${NC}"
echo -e "${RED}Failed:                      $total_failed${NC}"
echo -e "${YELLOW}Skipped (corrupted/unreadable): $total_skipped${NC}"
echo ""

if [ $total_processed -gt 0 ]; then
    echo -e "${GREEN}✓ Results saved to: $OUTPUT_CSV${NC}"
    echo ""
    echo "Preview of results:"
    echo "----------------------------------------"
    
    # Show first few lines of CSV
    if command -v column &> /dev/null; then
        head -n 6 "$OUTPUT_CSV" | column -t -s,
    else
        head -n 6 "$OUTPUT_CSV"
    fi
    
    echo "----------------------------------------"
    echo ""
    echo "Total entries in CSV: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
else
    echo -e "${RED}✗ No checkpoints were successfully processed${NC}"
fi

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="
