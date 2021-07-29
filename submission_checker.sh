
#!/bin/bash
# From https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/201047
# Requires "pip install csvkit" to handle commas in descriptions

SLUG=${1:-commonlitreadabilityprize} # default follows :-
for i in $(seq 1 108); do
    date
    TXT=$(kaggle competitions submissions -c $SLUG -v)
    if ! (echo "$TXT" | csvcut --columns=4 | grep -q pending); then
        kaggle competitions submissions -c $SLUG  # show subs
        # say nothing pending
        break
    fi
    sleep 119
done

