# Random Agents

## RandomAgent

Total solved: 0
Total failed: 10000
Average moves: 3.05
Max moves: 7
Min moves: 2

This is a fully random agent. It's not particularly expected to actually be good at the 8 queens problem. It just selects a random row and a random column.

The math here on if it can do a good job is interesting.

We are placing 8 queens on an 8X8 board.

There are 92 unique solutions to the problem. However, we are not generating the solution at once to the problem we are generating a pathway to the solution. For each of these solutions there are several unique pathways. Basically, each of these breaks out into several different paths to one solution, so the first placement can be any of the 8 locations, the second placement can be any of the 7 remaining locations, the next placement can be any of the remaining 6 locations etc. So basically for each solution there are 8! ways to get to that solution.

The total number of pathways that could lead to a solution randomly are `92×8!=92×40,320=3,709,440`. Which is great news that's a huge number.

You'd think we could find one. Unfortunately we basically can't. This is because the configurations available on a chess board are a bit nightmarish. This is because the number of possible queen placements, with no logic or constraints, is absolutely massive. Which means that there are 64 locations for the first piece, 63 locations for the second piece, 62 locations for the 3rd piece, etc to 8. Which... yikes is something like `2.3481972e+14` Which given we've gone scientific is a lot bigger.

This gives us about a `0.00000157969%` chance of randomly finding a solution... which is not ideal.

## RandomAgentByRow

Total solved: 0
Total failed: 10000
Average moves: 3.19
Max moves: 6
Min moves: 2

I tested an agent that is better at what it does "technically". It chooses a valid row at all times. The way it does this is by running through the rows from 0 - 8 in order and placing a queen in the row.

It's not meaningfully better at it in practice than the fully random. It has a tendency to make it slightly further on average. 3.19 moves instead of 3.01 or 3.10 instead of 3.04 moves. It's certainly not actually winning.

This does have some impact on the probability. We know that we can place in any of 8 spaces in the first row, then any of 8 spaces in the second row, etc. This means that there are `8^8` placements available to us. This is possible because we are not putting any effort in regarding column collisions at this point. This leaves us with `16,777,216` placement posibilities.

However, because we are now placing in a defined order (by row). Each of the 92 solutions can actually only have one possible placement order in this new system. So the full probability is now `92/16,777,216` or `0.00054836273%` Which great news this is almost a 350% improvement on our previous probability. Which is pretty great.

But unfortunately we still have 0 instances of the right answer.

## RandomAgentAlsoByColumn

This agent did appear to be immediately better as it revealed a bug in the previous agents when placing the final queen. Indicating that it was actually attempting to do that, whereas the agent before it may not have been. The question of course being is it enough better.

The probability space has been lowered again. We are now placing in the rows sequentially, and then only placing in columns that have not already been used. Which means that there are 8 options in the first row followed by 7 options in the next row, followed by 6 options in the next row. Which means that we now have an option space of `8!` or `40,320`. This means the probability is now `0.22%` that we may actually find an answer.

This seems like if we tried hard enough we could get a solution here. Like the little train that could.

Total solved: 23
Total failed: 9977
Average moves: 3.47
Max moves: 8
Min moves: 2

And when we run it we see that result it's able to win 23 times out of 10K which is the exact expectation for this agent.

