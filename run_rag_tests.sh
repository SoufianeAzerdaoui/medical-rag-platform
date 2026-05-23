#!/bin/bash
# Quick start script for RAG comprehensive testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}==================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3 installed"
    
    # Check virtual environment
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        print_warning "Virtual environment not found"
        print_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    print_success "Virtual environment activated"
    
    # Check backend API
    if ! curl -s http://127.0.0.1:8000/health &> /dev/null; then
        print_warning "Backend API not responding"
        print_info "You should start it in another terminal:"
        echo -e "${YELLOW}  cd $PROJECT_ROOT${NC}"
        echo -e "${YELLOW}  source venv/bin/activate${NC}"
        echo -e "${YELLOW}  python3 backend_api.py${NC}"
        echo ""
        read -p "Press ENTER once backend is started, or Ctrl+C to cancel..."
    else
        print_success "Backend API is responding"
    fi
}

# Run tests
run_tests() {
    print_header "Running RAG System Tests"
    
    cd "$PROJECT_ROOT"
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        TEST_MODE="all"
    else
        TEST_MODE="$1"
    fi
    
    case "$TEST_MODE" in
        quick)
            print_info "Running quick test suite (Tier 1 - 5 min)..."
            python3 scripts/evaluation/comprehensive_rag_tester.py \
                --suites suite_1_reference_ranges \
                         suite_2_single_analyte_lookup \
                         suite_7_document_filtering \
                         suite_9_safety_guardrails \
                --output reports/quick_test_$(date +%Y%m%d_%H%M%S).json
            ;;
        full)
            print_info "Running full test suite (ALL suites - 30 min)..."
            python3 scripts/evaluation/comprehensive_rag_tester.py \
                --output reports/full_test_$(date +%Y%m%d_%H%M%S).json
            ;;
        safety)
            print_info "Running SAFETY tests only (CRITICAL)..."
            python3 scripts/evaluation/comprehensive_rag_tester.py \
                --suites suite_9_safety_guardrails \
                --output reports/safety_test_$(date +%Y%m%d_%H%M%S).json
            ;;
        *)
            print_info "Running full test suite (default)..."
            python3 scripts/evaluation/comprehensive_rag_tester.py \
                --output reports/full_test_$(date +%Y%m%d_%H%M%S).json
            ;;
    esac
}

# Analyze results
analyze_results() {
    print_header "Analyzing Results"
    
    # Get latest report
    LATEST_REPORT=$(ls -t reports/rag_test_report.json 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_REPORT" ]; then
        print_error "No test report found"
        return 1
    fi
    
    print_info "Analyzing: $LATEST_REPORT\n"
    
    python3 scripts/evaluation/analyze_test_results.py "$LATEST_REPORT"
    
    return 0
}

# Main menu
show_menu() {
    print_header "Medical RAG System - Comprehensive Tester"
    
    echo "Select test mode:"
    echo ""
    echo "  1) Quick Tests     (Tier 1 - 5 min)  - Daily health check"
    echo "  2) Full Tests      (ALL - 30 min)    - Comprehensive assessment"
    echo "  3) Safety Only     (1 min)           - Critical checks only"
    echo "  4) Analyze Latest  (2 min)           - Show most recent results"
    echo "  5) Compare Results (2 min)           - Compare two test runs"
    echo "  6) Exit"
    echo ""
    read -p "Choose option (1-6): " choice
    
    case $choice in
        1)
            run_tests quick
            analyze_results
            ;;
        2)
            run_tests full
            analyze_results
            ;;
        3)
            run_tests safety
            analyze_results
            ;;
        4)
            analyze_results
            ;;
        5)
            echo ""
            echo "Available test reports:"
            ls -t reports/*test*.json 2>/dev/null | head -5 | nl
            echo ""
            read -p "Compare with which report? (1-5 or full path): " compare_choice
            
            if [[ "$compare_choice" =~ ^[0-9]$ ]]; then
                OLD_REPORT=$(ls -t reports/*test*.json 2>/dev/null | head -5 | sed -n "${compare_choice}p")
            else
                OLD_REPORT="$compare_choice"
            fi
            
            if [ -f "$OLD_REPORT" ]; then
                LATEST=$(ls -t reports/*test*.json 2>/dev/null | head -1)
                python3 scripts/evaluation/analyze_test_results.py "$LATEST" --compare "$OLD_REPORT"
            else
                print_error "Report not found: $OLD_REPORT"
            fi
            ;;
        6)
            echo ""
            print_info "Exiting. Good luck with your tests!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
    
    echo ""
    read -p "Press ENTER to return to menu or Ctrl+C to exit..."
    clear
    show_menu
}

# Main execution
main() {
    clear
    
    # Check if running in interactive mode
    if [ -t 0 ]; then
        check_prerequisites
        show_menu
    else
        # Non-interactive mode - run full tests
        check_prerequisites
        run_tests full
        analyze_results
    fi
}

main "$@"
