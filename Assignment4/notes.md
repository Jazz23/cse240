Check if subgoal is already in rule set
Check that subgoal is not already in set of all considered rules

Avoiding loops: If a rule requires a -> b, but we're looking for b -> a, don't consider it
Look for rule that has no dependencies

Create dependency stack -> Check this stack when looking for new rules