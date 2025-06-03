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

# Reinforcement Agents

## SimpleRandomReinforcementAgent

This is a pretty simple reinforcement agent.

It's primary goal is to find a single solution to the problem.

The way that it does this is by attempting to generate paths and then build upon them. When it finds a path that it believes in it will keep attempting to follow that path as long as it can and try to expand it.

It does include a small amount of discovery. This exists to allow it to recover from choosing what could be an impossible path early on.

This agent works okay. It tends to improve the more you actually let it train.

For example here:

Training SimpleRandomReinforcementAgent(epsilon=0.10) 1000 cases
Finished training SimpleRandomReinforcementAgent(epsilon=0.10) 1000 cases
Running SimpleRandomReinforcementAgent(epsilon=0.10) 100 cases
Total solved: 0
Total failed: 100
Average moves: 6.27
Max moves: 9
Min moves: 2

Training SimpleRandomReinforcementAgent(epsilon=0.10) 5000 cases
Finished training SimpleRandomReinforcementAgent(epsilon=0.10) 5000 cases
Running SimpleRandomReinforcementAgent(epsilon=0.10) 100 cases
Total solved: 5
Total failed: 95
Average moves: 6.20
Max moves: 9
Min moves: 2

Training SimpleRandomReinforcementAgent(epsilon=0.10) 10000 cases
Finished training SimpleRandomReinforcementAgent(epsilon=0.10) 10000 cases
Running SimpleRandomReinforcementAgent(epsilon=0.10) 100 cases
Total solved: 32
Total failed: 68
Average moves: 6.46
Max moves: 10
Min moves: 2

It went from 0 -> 5 -> 32.

But it has a tendency to overfit to a broken path as seen here.

Training SimpleRandomReinforcementAgent(epsilon=0.10) 20000 cases
Finished training SimpleRandomReinforcementAgent(epsilon=0.10) 20000 cases
Running SimpleRandomReinforcementAgent(epsilon=0.10) 100 cases
Total solved: 5
Total failed: 95
Average moves: 6.69
Max moves: 10
Min moves: 2

Even though it trained twice as much it got stuck in what was obviously a quite bad path.

The orginal version of this agent on a particular seed returned `4.25 -> 4.77 -> 5.46 -> 6.09 -> 6.07 -> 6.14` steps across training sizes. It was never able to solve this seed, although it could solve others.

I attempted to sort the paths to collapse the number of paths but that made things a lot worse.

# DecayFailingPaths

This is a slight remix of the `SimpleRandomReinforcementAgent` it actually changes very little, it just makes a brief attempt to decay scores on entries that don't seem to be working.

On the particular instance I'm attempting at the moment it has a small but noticable effect. Moving the quality up slightly. The effect we are attempting to fix was the issue or overfitting in the original agent. Which can cause it to get stuck in failures relatively agressively. We know there are states that are broken and unsolvable. The goal is to eventually trigger it to ignore those states.

Running a prticularly bad seed on the original agent we got average moves of `4.25 -> 4.77 -> 5.46 -> 6.09 -> 6.07 -> 6.14` working our way up to 20K training cases. We see here that 10K is particularly bad, it seems to be quite stuck in an obviously poor path.

With the new agent we see `4.25 -> 5.66 -> 5.96 -> 6.26 -> 6.62 -> 6.56 -> 6.14`. This is able to get itself at least partially unstuck in some of the smaller areas. It does still tend to struggle quite a bit in the larger training cases.

This agent was attempting to shift scores to be poisitive then penalize them. This caused problems because the lowest score became equivalent to 0. While this doesn't have an obvious effect it could be confusing if you are attempting to compare to a path that hasn't run yet.

I changed the penalty logic to just subtract the number of failures to make it a bit less complex and to just make negative numbers even more negative. This did have an impact `4.41 -> 5.47 -> 5.03 -> 6.37 -> 6.08 -> 6.62`. This version is a lot less stable. It seems to be abandoning paths too early. I believe this was because subtracting 1 is basically the same as removing the last item from the path scorewise. This also managed to solve 8 times when trained 20K times.

To test again I swapped `-1` for a decay rate of `0.2` per failure. This slows down abandoning a path hopefully. `4.41 -> 5.90 -> 5.98 -> 6.44 -> 6.19 -> 6.44`. This one is starting to be significantly better. Although it still has some moments it manages to solve starting at 5K cases. Solving 8 at 5k, 20 at 10K, and 19 at 20k. This isn't as good as a semi-intelligent random, but it's moving in the right direction.

```
def _rebuild_scores(self):
        best_scores: defaultdict[float, list[list[tuple[int, int]]]] = defaultdict(list)
        for score, paths in self.best_options.items():
            for path in paths:
                penalty = (1 + self.failing_paths[tuple(path)]) * 0.2
                decayed_score = score - penalty
                best_scores[decayed_score].append(path)
        return best_scores
```

We know that there are a lot of options after any given placement that could be fail without the placement actually being faulty. So we still have the question of abandoning too early. To check for this I brought the penalty down to 0.1. This is significantly worse. It ends up capping the avg steps at 5.9 for this case and the max solves at 13. Solving only 3 on the 20k case. I tried several values between these values, and they all tended to end up worse overall than 0.2. This could be related to the specific case of course.

Raising it also seems to make it worse overall at least on this seed. What is interesting is that sometimes making the number of steps potentially worse sometimes still makes it a bit more "solvable". At a decay of 0.5 it manages to find a single solution at 1K which it has not managed at any other value I have tried. However, the overall scores are problematic `4.41 -> 5.90 -> 6.22 -> 5.68 -> 5.52 -> 6.05`. It was able to find an answer much earlier, but it ends up having a lot of difficult actually maintaining that answer effectively for some reason. It's not obvious why as the correct answer is not penalized.

I tried changing the punishment based on the length of the path

```
    def _rebuild_scores(self):
        best_scores: defaultdict[float, list[list[tuple[int, int]]]] = defaultdict(list)
        for score, paths in self.best_options.items():
            for path in paths:
                penalty = (
                    (1 + self.failing_paths[tuple(path)]) * 0.2 * (1 - len(path) / 8)
                )
                decayed_score = score - penalty
                best_scores[decayed_score].append(path)
        return best_scores
```

This will basically get more angry at something that is failing super early. If you fail on step 2 that's significantly worse than failing on step 7.

This gives an interesting result `4.41 -> 5.90 -> 6.21 -> 6.79 -> 6.29 -> 6.11` Solving 11, 10, and 10 cases at 5K, 10K, and 20K respectively.

It's worth asking ourselves given that this actually lowers the penalties do we also need to up the decay factor to make up for it. This doesn't seem to obviously be the case as that make 10k jump up to 23 solutions, but it makes 20k go down to 3.

Shockingly, what actually works quite well here is to pull out the additional decay factor entirely.

```
    def _rebuild_scores(self):
        best_scores: defaultdict[float, list[list[tuple[int, int]]]] = defaultdict(list)
        for score, paths in self.best_options.items():
            for path in paths:
                penalty = (1 + self.failing_paths[tuple(path)]) * (1 - len(path) / 8)
                decayed_score = score - penalty
                best_scores[decayed_score].append(path)
        return best_scores
```

This gives us scores of `4.41 -> 5.90 -> 6.28 -> 6.49 -> 6.62 -> 6.03`. Which is still having some issues with too much training. However, it managed to solve 25 cases with only 1K trainign runs. And 41, 38, and 25 respectively in the higher runs. This is one of the most successful versions we have seen so far. Even though additional training makes it slightly worse.

