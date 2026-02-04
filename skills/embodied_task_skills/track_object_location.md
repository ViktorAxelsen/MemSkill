# Track Object Location

## Description

Memory management skill for explicitly tracking the locations and states of objects necessary for task completion.

## Purpose

Explicitly track the location and state of an object necessary for task completion.

## When to Use

- The text chunk mentions an object's location or state.
- The object's location or state is crucial for future task steps.

## How to Apply

- Identify the object, its location, and relevant state from the text chunk.
- Create a new memory item with the object-location-state triplet.

## Constraints

- Only track locations and states of objects relevant to the task.
- Update existing location memories if new information is provided.

Action type: INSERT only.
